mkdir -p ~/.streamlit/
echo "[general]
email = \"ankita.prakash1998@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
