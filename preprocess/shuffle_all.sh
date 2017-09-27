a=/home/jjery2243542/datasets/summary/cnn_dailymail/processed/no_punc/content/train.txt
b=/home/jjery2243542/datasets/summary/cnn_dailymail/processed/no_punc/title/train.txt
c=/home/jjery2243542/datasets/summary/cnn_dailymail/processed/shuffle/content/train.txt
d=/home/jjery2243542/datasets/summary/cnn_dailymail/processed/shuffle/title/train.txt
python3 shuffle.py $a $b $c $d
