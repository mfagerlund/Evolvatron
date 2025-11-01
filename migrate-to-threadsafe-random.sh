#!/bin/bash

# Script to migrate from Random to ThreadSafeRandom

cd /c/Dev/Evolvatron

# Function to process a single file
process_file() {
    local file="$1"

    # Add using statement if not already present
    if ! grep -q "using Colonel.Framework.Utilities;" "$file"; then
        sed -i '1i using Colonel.Framework.Utilities;\n' "$file"
    fi

    # Replace random method calls
    sed -i 's/random\.NextSingle()/ThreadSafeRandom.GetNext01Float()/g' "$file"
    sed -i 's/random\.Next(/ThreadSafeRandom.GetNextInt(/g' "$file"

    # Replace new Random(seed) with ThreadSafeRandom.SetRandomSeed(seed)
    sed -i 's/_random = new Random(\([^)]*\));/ThreadSafeRandom.SetRandomSeed(\1);/g' "$file"
    sed -i 's/new Random(\([^)]*\))/ThreadSafeRandom.SetRandomSeed(\1)/g' "$file"

    # Remove Random field declarations
    sed -i '/private readonly Random _random;/d' "$file"
    sed -i '/private Random _random;/d' "$file"

    # Remove Random random parameters (simple cases)
    sed -i 's/, Random random)/)/g' "$file"
    sed -i 's/(Random random)/()/g' "$file"
    sed -i 's/(Random random, /(/g' "$file"
}

# Process all CS files in Evolvion
echo "Processing Evolvion files..."
find Evolvatron.Evolvion -name "*.cs" -type f | while read f; do
    echo "  Processing: $f"
    process_file "$f"
done

echo "Migration complete!"
