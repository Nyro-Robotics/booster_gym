# For ~/.bashrc file
```bash
alias start-control='f() { 
    url="$1"; 
    proto="${url%%://*}"; 
    rest="${url#*://}"; 
    host="${rest%%:*}"; 
    port="${rest##*:}"; 
    echo "🔍 Debug: URL=$url, Protocol=$proto, Host=$host, Port=$port"; 
    conda activate nzbooster; 
    if [[ "$proto" == "ws" || "$proto" == "wss" ]]; then 
        echo "🌐 Using WebSocket communication mode"; 
        python test/deploy.py 127.0.0.1 --communication websocket --ws-host "$host" --ws-port "$port"; 
    elif [[ "$proto" == "tcp" ]]; then 
        echo "⚡ Using ZeroMQ communication mode"; 
        python test/deploy.py 127.0.0.1 --communication zeromq --zmq-address "$url"; 
    else 
        echo "❌ Unsupported protocol: $proto. Use ws://, wss://, or tcp://"; 
    fi; 
}; f'

alias sim2real="conda activate fc_booster && cd /home/master/booster_gym/test/sim2real && python rl_policy/loco_manip/loco_manip.py --config=config/t1_29dof.yaml --model_path=models/t1_29dof.onnx"

alias test-zeromq='f() { url="$1"; if [[ "$url" =~ ^tcp:// ]]; then python test/example_zeromq_client.py --address "$url"; else echo "❌ Please provide a TCP URL in format: tcp://host:port"; fi; }; f'
alias test-websocket='f() { url="$1"; proto="${url%%://*}"; rest="${url#*://}"; host="${rest%%:*}"; port="${rest##*:}"; python test/example_websocket_client.py --host "$host" --port "$port"; } 

```