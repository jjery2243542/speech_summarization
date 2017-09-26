# $1: input directory with raw content and title
# $2: output directory with tokenized content and title
# $3: output directory with split train/dev/test set
input_dir=~/datasets/cnn_news/raw_data/
token_dir=~/datasets/cnn_news/processed/tokenized/
split_dir=~/datasets/cnn_news/processed/split/
condition_dir=~/datasets/cnn_news/processed/conditioned/
depunc_dir=~/datasets/cnn_news/processed/depunc/
tmp_dir=/tmp/corenlp_tmp

# tokenize
./tokenizer.sh $input_dir/content.txt 2>/dev/null 
python3 merge_xml.py $tmp_dir $token_dir/content.txt
./tokenizer.sh $input_dir/title.txt 2>/dev/null
python3 merge_xml.py $tmp_dir $token_dir/title.txt
cp $input_dir/index.txt $token_dir/index.txt
# depunc
python3 depunc.py $token_dir/title.txt $depunc_dir/title.txt
cp $token_dir/content.txt $depunc_dir/content.txt
cp $token_dir/index.txt $depunc_dir/index.txt
# set condition
python3 condition.py $depunc_dir $condition_dir
# split to train, valid, test
python3 split.py $condition_dir $split_dir 0.8,0.1,0.1
