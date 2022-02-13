while read -r line; do
    arr=($line)
    mkdir -p $(dirname ${arr[0]})
done < download_links

while read -r line; do
    arr=($line)
    echo Downloading ${arr[0]} from ${arr[1]}
    if ! [[ -e "${arr[0]}" ]]; then
        curl -L -o ${arr[0]} ${arr[1]}
        echo Downloading ${arr[0]} Done
    else
        echo ${arr[0]} File already exists
    fi
done < download_links
