// int8 weights times float32 activations, one workgroup per output value.
//
// WGSL has no 8-bit type, so each row arrives packed four int8 to a u32.
// extractBits on an i32 sign-extends, and int8 -> f32 is exact, so the
// shader multiplies exactly the integers the CPU kernel does. The scale is
// applied once per row to the finished sum, as on the CPU.
//
// The order of the additions is not the CPU's: 64 threads each sum a strided
// slice, then a tree folds the 64 partial sums. Float addition does not
// associate, so GPU and CPU agree to float32 tolerance, not bitwise.

struct Params {
    rows: u32,           // rows in this chunk of the matrix
    words_per_row: u32,  // in_features / 4, after padding to a multiple of 4
    tokens: u32,
    rows_per_slice: u32, // workgroups along x; rows wrap into y past this
    row_offset: u32,     // where this chunk's rows sit in the whole output
    total_rows: u32,     // output features of the whole matrix
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const LANES: u32 = 64u;
var<workgroup> partial: array<f32, LANES>;

fn unpack(word: u32) -> vec4<f32> {
    let w = bitcast<i32>(word);
    return vec4<f32>(
        f32(extractBits(w, 0u, 8u)),
        f32(extractBits(w, 8u, 8u)),
        f32(extractBits(w, 16u, 8u)),
        f32(extractBits(w, 24u, 8u)),
    );
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // WebGPU caps a dispatch dimension at 65535 workgroups and the LM head
    // has 151936 rows, so rows run along x and wrap into y.
    let row = group.x + group.y * params.rows_per_slice;
    let token = group.z;
    // Uniform across the workgroup (it depends only on workgroup_id), so
    // returning here cannot strand a thread at the barrier below.
    if (row >= params.rows) {
        return;
    }

    let w0 = row * params.words_per_row;
    let x0 = token * params.words_per_row;

    var sum = 0.0;
    for (var i = lane; i < params.words_per_row; i += LANES) {
        sum += dot(unpack(weights[w0 + i]), x[x0 + i]);
    }
    partial[lane] = sum;
    workgroupBarrier();

    for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        out[token * params.total_rows + params.row_offset + row] = partial[0] * scales[row];
    }
}
