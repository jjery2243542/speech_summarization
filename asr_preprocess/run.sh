# $1: input directory with raw content and title
# $2: output directory with tokenized content and title
# $3: output directory with split train/dev/test set
input_dir=~/datasets/cnn_news/raw_data/
token_dir=~/datasets/cnn_news/processed/tokenized/
split_dir=~/datasets/cnn_news/processed/split/
tmp_dir=/tmp/corenlp_tmp
tmp2_dir=/tmp/preprocess_tmp

rm -rf $tmp2_dir
mkdir $tmp2_dir

./tokenizer.sh $input_dir/content.txt 2>/dev/null 
python3 merge_xml.py $tmp_dir $token_dir/content.txt
./tokenizer.sh $input_dir/title.txt 2>/dev/null
python3 merge_xml.py $tmp_dir $token_dir/title.txt
cp $input_dir/index.txt $token_dir/index.txt
python3 condition.py $token_dir $tmp2_dir
python3 split.py $tmp2_dir $split_dir 0.8,0.1,0.1
