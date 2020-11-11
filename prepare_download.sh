cd 
mkdir downloads
mkdir downloads_zip
cp -R work/* downloads/
cd downloads
find ./ -type l -exec sh -c 'readlink -f "{}" 1>/dev/null || rm "{}"' -- "{}" \;
cd ..
tar -hcf - downloads | tar -xf - -C downloads_zip
tar cvfz downloads_zip.tar.gz  'downloads_zip' 
split -b 90m downloads_zip.tar.gz downloads_zip.tar.gz.par
mv downloads_zip.tar.gz.par* work/
