depunc_dir=$1
for filename in $depunc_dir/*; do
    python3 depunc.py $filename $filename.nopunc
done

