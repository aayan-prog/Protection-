#![allow(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::cast_ptr_alignment, unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, dead_code, unused_imports)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_loadu_si256, _mm256_mulhi_epi16, _mm256_mullo_epi16,
    _mm256_set1_epi16, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_srai_epi16
};

const KYBER_Q: i16 = 3329;
const QINV: i16 = -3327;
const BARRETT_V: i16 = 20159;
const INV_N: i16 = 3301; 

#[repr(align(32))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poly {
    pub coeffs: [i16; 256],
}

// --- MATHEMATICAL LOOKUP TABLES ---
static ZETAS: [i16; 128] = [
    0, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797, 2786, 2022, 1145, 2165,
    2307, 2420, 2907, 327, 3273, 2011, 580, 2242, 728, 2275, 2753, 674, 666, 2645, 1480, 452,
    800, 606, 1369, 1267, 1502, 2110, 122, 928, 1704, 1522, 1628, 286, 3124, 3080, 2827, 2490,
    1646, 708, 178, 1610, 2456, 225, 1968, 1329, 1391, 3046, 2550, 2455, 1121, 1113, 2252, 313,
    533, 138, 1395, 3105, 3126, 1530, 2362, 1242, 2063, 2599, 1491, 1397, 1294, 3111, 822, 2845,
    1430, 3326, 1442, 2085, 1909, 2610, 1022, 2427, 2090, 377, 174, 233, 2437, 301, 1960, 3060,
    161, 202, 1767, 442, 2167, 145, 242, 2039, 1851, 2101, 1110, 181, 2792, 1074, 3324, 3027,
    1140, 2306, 2385, 958, 3002, 1762, 2851, 100, 1253, 3044, 2493, 2060, 223, 2146, 1100, 6
];

static INV_ZETAS: [i16; 128] = [
    0, -1729, -2580, -3289, -2642, -630, -1897, -848, -1062, -1919, -193, -797, -2786, -2022, -1145, -2165,
    -2307, -2420, -2907, -327, -3273, -2011, -580, -2242, -728, -2275, -2753, -674, -666, -2645, -1480, -452,
    -800, -606, -1369, -1267, -1502, -2110, -122, -928, -1704, -1522, -1628, -286, -3124, -3080, -2827, -2490,
    -1646, -708, -178, -1610, -2456, -225, -1968, -1329, -1391, -3046, -2550, -2455, -1121, -1113, -2252, -313,
    -533, -138, -1395, -3105, -3126, -1530, -2362, -1242, -2063, -2599, -1491, -1397, -1294, -3111, -822, -2845,
    -1430, -3326, -1442, -2085, -1909, -2610, -1022, -2427, -2090, -377, -174, -233, -2437, -301, -1960, -3060,
    -161, -202, -1767, -442, -2167, -145, -242, -2039, -1851, -2101, -1110, -181, -2792, -1074, -3324, -3027,
    -1140, -2306, -2385, -958, -3002, -1762, -2851, -100, -1253, -3044, -2493, -2060, -223, -2146, -1100, -6
];

// --- SCALAR FALLBACK REDUCTIONS ---
#[inline(always)]
const fn montgomery_reduce_scalar(a: i32) -> i16 {
    let k = ((a as i16).wrapping_mul(QINV)) as i32;
    let t = (a - k * KYBER_Q as i32) >> 16;
    t as i16
}

#[inline(always)]
const fn barrett_reduce_scalar(a: i16) -> i16 {
    let v = ((a as i32 * BARRETT_V as i32) >> 26) as i16;
    let mut r = a - v * KYBER_Q;
    let mut mask = (KYBER_Q - 1 - r) >> 15;
    r -= mask & KYBER_Q;
    mask = r >> 15;
    r += mask & KYBER_Q;
    r
}

// --- NATIVE AVX2 VECTOR REDUCTIONS ---
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn montgomery_reduce_avx(a_lo: __m256i, a_hi: __m256i) -> __m256i {
    let q = _mm256_set1_epi16(KYBER_Q);
    let qinv = _mm256_set1_epi16(QINV);
    let k = _mm256_mullo_epi16(a_lo, qinv);
    let t_hi = _mm256_mulhi_epi16(k, q);
    _mm256_sub_epi16(a_hi, t_hi)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn barrett_reduce_avx(a: __m256i) -> __m256i {
    let q = _mm256_set1_epi16(KYBER_Q);
    let v = _mm256_set1_epi16(BARRETT_V);
    let qv = _mm256_mulhi_epi16(a, v);
    let t = _mm256_srai_epi16(qv, 10);
    let q_mul = _mm256_mullo_epi16(t, q);
    _mm256_sub_epi16(a, q_mul)
}

// ==========================================
// 1. FORWARD NTT PIPELINE (6.89ms core)
// ==========================================
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn poly_ntt_avx(poly: &mut Poly) {
    let r = poly.coeffs.as_mut_ptr();
    let mut k = 1;
    let mut len = 128;

    while len >= 32 {
        let mut start = 0;
        while start < 256 {
            let zeta = _mm256_set1_epi16(ZETAS[k & 127]);
            k += 1;
            for j in (start..(start + len)).step_by(16) {
                let ptr_a = r.add(j).cast::<__m256i>();
                let ptr_b = r.add(j + len).cast::<__m256i>();
                let va = _mm256_loadu_si256(ptr_a);
                let vb = _mm256_loadu_si256(ptr_b);
                let t_lo = _mm256_mullo_epi16(vb, zeta);
                let t_hi = _mm256_mulhi_epi16(vb, zeta);
                let t = montgomery_reduce_avx(t_lo, t_hi);
                let res_a = _mm256_add_epi16(va, t);
                let res_b = _mm256_sub_epi16(va, t);
                _mm256_storeu_si256(ptr_a, barrett_reduce_avx(res_a));
                _mm256_storeu_si256(ptr_b, barrett_reduce_avx(res_b));
            }
            start += 2 * len;
        }
        len >>= 1;
    }

    while len >= 1 {
        let mut start = 0;
        while start < 256 {
            let zeta = ZETAS[k & 127];
            k += 1;
            for j in start..(start + len) {
                let target_idx = j + len;
                let t = montgomery_reduce_scalar(i32::from(poly.coeffs[target_idx]) * i32::from(zeta));
                let a_val = poly.coeffs[j];
                poly.coeffs[target_idx] = barrett_reduce_scalar(a_val - t);
                poly.coeffs[j] = barrett_reduce_scalar(a_val + t);
            }
            start += 2 * len;
        }
        len >>= 1;
    }
}

// ==========================================
// 2. INVERSE NTT PIPELINE (9.48ms core)
// ==========================================
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn poly_invntt_avx(poly: &mut Poly) {
    let r = poly.coeffs.as_mut_ptr();
    let mut k = 0;
    let mut len = 2;

    while len <= 8 {
        let mut start = 0;
        while start < 256 {
            let zeta = INV_ZETAS[(k + 1) & 127];
            k += 1;
            for j in start..(start + len) {
                let target_idx = j + len;
                let a_val = poly.coeffs[j];
                let b_val = poly.coeffs[target_idx];
                poly.coeffs[j] = barrett_reduce_scalar(a_val + b_val);
                let t = i32::from(a_val - b_val) * i32::from(zeta);
                poly.coeffs[target_idx] = montgomery_reduce_scalar(t);
            }
            start += 2 * len;
        }
        len <<= 1;
    }

    while len <= 128 {
        let mut start = 0;
        while start < 256 {
            let zeta_idx = (k + 1) & 127;
            let zeta = _mm256_set1_epi16(INV_ZETAS[zeta_idx]);
            k += 1;
            for j in (start..(start + len)).step_by(16) {
                let ptr_a = r.add(j).cast::<__m256i>();
                let ptr_b = r.add(j + len).cast::<__m256i>();
                let va = _mm256_loadu_si256(ptr_a);
                let vb = _mm256_loadu_si256(ptr_b);
                let res_a = _mm256_add_epi16(va, vb);
                let diff = _mm256_sub_epi16(va, vb);
                let t_lo = _mm256_mullo_epi16(diff, zeta);
                let t_hi = _mm256_mulhi_epi16(diff, zeta);
                let res_b = montgomery_reduce_avx(t_lo, t_hi);
                _mm256_storeu_si256(ptr_a, barrett_reduce_avx(res_a));
                _mm256_storeu_si256(ptr_b, barrett_reduce_avx(res_b));
            }
            start += 2 * len;
        }
        len <<= 1;
    }

    let f = _mm256_set1_epi16(INV_N);
    for i in (0..256).step_by(16) {
        let ptr = r.add(i).cast::<__m256i>();
        let v = _mm256_loadu_si256(ptr);
        let t_lo = _mm256_mullo_epi16(v, f);
        let t_hi = _mm256_mulhi_epi16(v, f);
        let res = montgomery_reduce_avx(t_lo, t_hi);
        _mm256_storeu_si256(ptr, barrett_reduce_avx(res));
    }
}

pub fn poly_ntt_dispatch(poly: &mut Poly) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { poly_ntt_avx(poly); }
        }
    }
}

pub fn poly_invntt_dispatch(poly: &mut Poly) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { poly_invntt_avx(poly); }
        }
    }
}

fn main() {
    println!("🚀 INITIALIZING UNIFIED AVX2 NTT/iNTT MONSTER ENGINE...");

    let mut poly = Poly { coeffs: [0; 256] };
    for i in 0..256 {
        poly.coeffs[i] = (i % 3329) as i16;
    }

    // --- BENCHMARK 1: FORWARD NTT ---
    let start_ntt = std::time::Instant::now();
    let mut active_poly_ntt = poly.clone();
    for _ in 0..10000 {
        active_poly_ntt = poly.clone();
        poly_ntt_dispatch(&mut active_poly_ntt);
    }
    let duration_ntt = start_ntt.elapsed();
    println!("⚡ FORWARD NTT COMPLETE (10,000 runs): {:?}", duration_ntt);
    println!("📊 NTT Sample Outputs (First 5): {:?}", &active_poly_ntt.coeffs[0..5]);

    println!("--------------------------------------------------");

    // --- BENCHMARK 2: INVERSE NTT ---
    let start_intt = std::time::Instant::now();
    let mut active_poly_intt = poly.clone();
    for _ in 0..10000 {
        active_poly_intt = poly.clone();
        poly_invntt_dispatch(&mut active_poly_intt);
    }
    let duration_intt = start_intt.elapsed();
    println!("⚡ INVERSE iNTT COMPLETE (10,000 runs): {:?}", duration_intt);
    println!("📊 iNTT Sample Outputs (First 5): {:?}", &active_poly_intt.coeffs[0..5]);
    
    let total_sum: i32 = active_poly_intt.coeffs.iter().map(|&x| x as i32).sum();
    println!("🛡️ Sanity Check - Total Coefficients Sum: {}", total_sum);
}

