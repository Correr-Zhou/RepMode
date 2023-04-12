declare -a arr=("beta_actin" "myosin_iib" "membrane_caax_63x" "desmoplakin" "sec61_beta" "st6gal1" "fibrillarin" "lamin_b1" "alpha_tubulin" "tom20" "zo1")

for i in "${arr[@]}"
do
   wget -c https://s3-us-west-2.amazonaws.com/downloads.allencell.org/publication-data/label-free-prediction/$i.tar.gz
   tar -C ./data -xvzf ./$i.tar.gz
done
