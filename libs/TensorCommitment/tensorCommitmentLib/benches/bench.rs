use std::time::Duration;

use ark_bn254::{Bn254, Fr};
use ark_ff::UniformRand;
use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, Criterion};
use mmp::MMP;
use num_integer::binomial;

fn bench_setup(c: &mut Criterion) {
    let rng = &mut test_rng();
    let mut group = c.benchmark_group("setup");
    group.sample_size(10);

    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
        for m in [1, 2, 3, 4, 5] {
            for d in [1, 2, 3, 4, 5] {
                group.bench_function(&format!("setup n: {} m: {} d: {}", n, m, d), |b| {
                    b.iter(|| MMP::<Bn254>::setup(rng, m, n, d))
                });
            }
        }
    }
    group.finish();
}

fn bench_commit(c: &mut Criterion) {
    let rng = &mut test_rng();
    let mut group = c.benchmark_group("commit");
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(500));

    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
        for m in [1, 2, 3, 4, 5] {
            for d in [2, 3, 4, 5, 6] {
                let (ck, _vk) = MMP::<Bn254>::setup(rng, m, n, d);

                let fs = (0..n)
                    .map(|_| {
                        (0..binomial(m + d - 1, d - 1))
                            .map(|_| Fr::rand(rng))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let _v = (0..m).map(|_| Fr::rand(rng)).collect::<Vec<_>>();

                group.bench_function(&format!("commit n: {} m: {} d: {}", n, m, d - 1), |b| {
                    b.iter(|| MMP::<Bn254>::commit(&ck, &fs))
                });
            }
        }
    }
    group.finish();
}

fn bench_prove(c: &mut Criterion) {
    let rng = &mut test_rng();
    let mut group = c.benchmark_group("prove");
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(500));

    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
        for m in [1, 2, 3, 4, 5] {
            for d in [2, 3, 4, 5, 6] {
                let (ck, _vk) = MMP::<Bn254>::setup(rng, m, n, d);

                let fs = (0..n)
                    .map(|_| {
                        (0..binomial(m + d - 1, d - 1))
                            .map(|_| Fr::rand(rng))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let v = (0..m).map(|_| Fr::rand(rng)).collect::<Vec<_>>();

                let (c_pst, c_afg) = MMP::<Bn254>::commit(&ck, &fs);

                group.bench_function(&format!("prove n: {} m: {} d: {}", n, m, d - 1), |b| {
                    b.iter(|| MMP::<Bn254>::prove(d, &ck, &fs, &c_pst, &c_afg, &v))
                });
            }
        }
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let rng = &mut test_rng();
    let mut group = c.benchmark_group("verify");
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(500));

    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
        for m in [1, 2, 3, 4, 5] {
            for d in [2, 3, 4, 5, 6] {
                let (ck, vk) = MMP::<Bn254>::setup(rng, m, n, d);

                let fs = (0..n)
                    .map(|_| {
                        (0..binomial(m + d - 1, d - 1))
                            .map(|_| Fr::rand(rng))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let v = (0..m).map(|_| Fr::rand(rng)).collect::<Vec<_>>();

                let (c_pst, c_afg) = MMP::<Bn254>::commit(&ck, &fs);

                let (c_ped, pi) = MMP::<Bn254>::prove(d, &ck, &fs, &c_pst, &c_afg, &v);

                group.bench_function(&format!("verify n: {} m: {} d: {}", n, m, d - 1), |b| {
                    b.iter(|| MMP::<Bn254>::verify(&vk, c_afg, c_ped, &v, &pi))
                });

                assert!(MMP::<Bn254>::verify(&vk, c_afg, c_ped, &v, &pi));
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_setup,
    bench_commit,
    bench_prove,
    bench_verify
);
criterion_main!(benches);
