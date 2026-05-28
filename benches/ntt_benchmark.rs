use criterion::{criterion_group, criterion_main, Criterion};
use kyber_ntt_hardened::{Poly, poly_ntt_scalar}; 

fn bench_ntt(c: &mut Criterion) {
    let mut poly = Poly { coeffs: [0; 256] };
    for i in 0..256 { 
        poly.coeffs[i] = i as i16; 
    }

    c.bench_function("Kyber NTT Speed Test", |b| {
        b.iter(|| {
            let mut p = poly.clone();
            poly_ntt_scalar(&mut p);
        })
    });
}

criterion_group!(benches, bench_ntt);
criterion_main!(benches);
