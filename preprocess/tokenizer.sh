# this script is aimed to tokenize text file
# input: un-tokenized file, each line is a document
# output: tokenized file, each line is a document

tmp_dir=/tmp/corenlp_tmp/

rm -rf $tmp_dir
mkdir $tmp_dir
# dump each line to separate files
filename=$1
index_file="index.txt"
i=0
while read line; do
    filename=`printf %06d.txt $i`
    echo $tmp_dir/$filename >> $tmp_dir/$index_file
    echo $line > $tmp_dir/$filename

    i=$(( $i + 1 ))
done < $filename

corenlp.sh -annotators tokenize,ssplit,pos,lemma -filelist $tmp_dir/$index_file -outputDirectory $tmp_dir 
