//! The int8 matmul on the GPU, through WebGPU.
//!
//! Phase 7 was scoped as CPU SIMD first, then compute shaders. The reason for
//! WebGPU rather than CUDA or Metal is the hardware: the reference laptop's
//! GPU is an Intel Iris Xe, and `wgpu` reaches it -- and any reviewer's GPU --
//! through Vulkan, Metal or DX12 without a vendor toolkit.
//!
//! The shape of the problem does not change on a GPU. Decode is a matvec over
//! every weight in the model, so it is bound by how fast the weights can be
//! read. The one thing that would make it hopeless is moving them across the
//! bus every token, so a [`GpuMatrix`] is uploaded once and stays resident;
//! only the activations go up and the outputs come back.
//!
//! This crate is separate from `rust/` on purpose. The CPU kernels have one
//! dependency and build in seconds; wgpu is a graphics stack. Nothing that
//! uses the CPU kernels should have to compile it.

use wgpu::util::DeviceExt;

/// WebGPU's per-dimension cap on workgroups in one dispatch.
const MAX_GROUPS_PER_DIM: u32 = 65_535;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    rows: u32,
    words_per_row: u32,
    tokens: u32,
    rows_per_slice: u32,
    row_offset: u32,
    total_rows: u32,
    _pad: [u32; 2],
}

/// A device, a queue and the compiled int8 matmul pipeline.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    info: wgpu::AdapterInfo,
}

/// An int8 weight matrix resident in GPU memory, uploaded once.
///
/// Held as row chunks, each small enough for one storage binding. A device
/// may cap a binding at 128 MiB -- Mesa's llvmpipe does -- and the LM head is
/// 151936 x 896 int8 = 130 MiB, so one buffer cannot be assumed to fit.
pub struct GpuMatrix {
    chunks: Vec<Chunk>,
    rows: usize,
    in_features: usize,
    words_per_row: usize,
}

struct Chunk {
    weights: wgpu::Buffer,
    scales: wgpu::Buffer,
    row_offset: usize,
    rows: usize,
}

impl GpuMatrix {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn in_features(&self) -> usize {
        self.in_features
    }
}

impl Gpu {
    /// The first adapter wgpu offers, preferring a real GPU.
    ///
    /// On a machine with none, a software Vulkan driver such as Mesa's
    /// llvmpipe still answers: right for checking the shader's arithmetic,
    /// meaningless for timing it. [`Gpu::is_software`] says which you got.
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("nanoinfer"),
                // The LM head is 151936 x 896 int8 = 136 MB, past the 128 MB
                // a binding may hold by default; ask for what the adapter has.
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("could not open the device: {e}"))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_i8"),
            source: wgpu::ShaderSource::Wgsl(include_str!("matmul_i8.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_i8"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self { device, queue, pipeline, info })
    }

    /// The adapter's name and backend, for benchmarks to record.
    pub fn describe(&self) -> String {
        format!("{} ({:?}, {:?})", self.info.name, self.info.backend, self.info.device_type)
    }

    /// True on a CPU-emulated adapter, where timings mean nothing.
    pub fn is_software(&self) -> bool {
        self.info.device_type == wgpu::DeviceType::Cpu
    }

    /// Upload a row-major `[rows, in_features]` int8 matrix and its per-row
    /// scales, the layout `quantize_rows_i8` produces.
    ///
    /// Rows are packed four int8 to a u32 and padded with zeros to a multiple
    /// of four; a zero weight adds nothing, so padding cannot change a sum.
    pub fn upload(&self, quantized: &[i8], scales: &[f32], in_features: usize) -> GpuMatrix {
        let limits = self.device.limits();
        let max_bytes = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        let row_bytes = (in_features.div_ceil(4) * 4) as u64;
        self.upload_chunked(quantized, scales, in_features, (max_bytes / row_bytes).max(1) as usize)
    }

    /// [`Gpu::upload`] with an explicit cap on rows per chunk, so tests can
    /// exercise several chunks without a matrix past the device's limit.
    #[doc(hidden)]
    pub fn upload_chunked(
        &self,
        quantized: &[i8],
        scales: &[f32],
        in_features: usize,
        max_rows_per_chunk: usize,
    ) -> GpuMatrix {
        let rows = scales.len();
        assert_eq!(quantized.len(), rows * in_features, "weight shape mismatch");
        assert!(max_rows_per_chunk > 0, "a chunk needs at least one row");
        let words_per_row = in_features.div_ceil(4);

        let init = |label, contents: &[u8]| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        let mut chunks = Vec::new();
        for row_offset in (0..rows).step_by(max_rows_per_chunk) {
            let chunk_rows = max_rows_per_chunk.min(rows - row_offset);
            let mut packed = vec![0u32; chunk_rows * words_per_row];
            for r in 0..chunk_rows {
                let src = &quantized[(row_offset + r) * in_features..(row_offset + r + 1) * in_features];
                let dst = &mut packed[r * words_per_row..(r + 1) * words_per_row];
                for (i, &q) in src.iter().enumerate() {
                    dst[i / 4] |= u32::from(q as u8) << (8 * (i % 4));
                }
            }
            chunks.push(Chunk {
                weights: init("weights", bytemuck::cast_slice(&packed)),
                scales: init("scales", bytemuck::cast_slice(&scales[row_offset..row_offset + chunk_rows])),
                row_offset,
                rows: chunk_rows,
            });
        }
        GpuMatrix { chunks, rows, in_features, words_per_row }
    }

    /// Number of row chunks a matrix was split into, for tests.
    #[doc(hidden)]
    pub fn chunk_count(matrix: &GpuMatrix) -> usize {
        matrix.chunks.len()
    }

    /// `x @ W.T` for `x` row-major `[tokens, in_features]`; returns
    /// `[tokens, rows]` row-major, the shape the NumPy engine expects.
    pub fn matmul(&self, matrix: &GpuMatrix, x: &[f32], tokens: usize) -> Vec<f32> {
        assert!(tokens > 0, "expected at least one token");
        assert_eq!(x.len(), tokens * matrix.in_features, "activation shape mismatch");

        // Activations padded to the same multiple of four as the weights.
        let width = matrix.words_per_row * 4;
        let mut padded = vec![0f32; tokens * width];
        for t in 0..tokens {
            padded[t * width..t * width + matrix.in_features]
                .copy_from_slice(&x[t * matrix.in_features..(t + 1) * matrix.in_features]);
        }

        let device = &self.device;
        let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: bytemuck::cast_slice(&padded),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_bytes = (tokens * matrix.rows * 4) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // One dispatch per chunk, all in one submission; each writes its own
        // rows of the shared output.
        let mut encoder = device.create_command_encoder(&Default::default());
        for chunk in &matrix.chunks {
            let rows = chunk.rows as u32;
            let rows_per_slice = rows.min(MAX_GROUPS_PER_DIM);
            let params = Params {
                rows,
                words_per_row: matrix.words_per_row as u32,
                tokens: tokens as u32,
                rows_per_slice,
                row_offset: chunk.row_offset as u32,
                total_rows: matrix.rows as u32,
                _pad: [0; 2],
            };
            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_i8"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: chunk.weights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: chunk.scales.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows_per_slice, rows.div_ceil(rows_per_slice), tokens as u32);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_bytes);
        self.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| result.expect("readback failed"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
        let view = slice.get_mapped_range().expect("readback not mapped");
        let out = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        readback.unmap();
        out
    }
}
