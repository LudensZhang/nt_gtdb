mkdir prokka_out/test_1000g
for file in test_genomes_1000g/*.fna; do
    base=$(basename "$file" .fna)
    prokka --outdir prokka_out/"$base" --prefix "$base" "$file" --cpus 64
done