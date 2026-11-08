# Write the Streamlit application to app.py first
with open("app.py", "w") as f:
    f.write('''import re
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Wireless Phone Camera",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Wireless Phone Camera")
st.write("Connect your smartphone camera to your laptop wirelessly.")

room_input = st.text_input(
    "Enter a private room code",
    placeholder="Example: amit-camera-4821"
)

room = re.sub(r"[^a-zA-Z0-9-]", "", room_input).lower()[:30]

role = st.radio(
    "Select device",
    ["Phone Camera", "Laptop Viewer"],
    horizontal=True
)

if len(room) < 4:
    st.info("Enter a room code containing at least 4 characters.")
    st.stop()

camera_id = f"camera-{room}"
viewer_id = f"viewer-{room}"

role_js = "phone" if role == "Phone Camera" else "laptop"

html = r"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">

<script src="https://unpkg.com/peerjs@1.5.4/dist/peerjs.min.js"></script>

<style>
body {
    font-family: Arial;
    padding: 10px;
}

video {
    width: 100%;
    max-height: 65vh;
    background: black;
    border-radius: 12px;
}

button {
    padding: 12px;
    margin: 5px;
    border: none;
    border-radius: 8px;
    background: #1769aa;
    color: white;
    font-size: 15px;
}

#status {
    margin: 15px 0;
    font-weight: bold;
}
</style>
</head>

<body>

<div id="phone" style="display:none">
    <h3>Phone Camera</h3>
    <video id="localVideo" autoplay playsinline muted></video>
    <br>
    <button id="start">Start Camera</button>
    <button id="stop">Stop Camera</button>
</div>

<div id="laptop" style="display:none">
    <h3>Live Phone Camera</h3>
    <video id="remoteVideo" autoplay playsinline controls></video>
    <br>
    <button id="connect">Connect to Phone</button>
    <button id="disconnect">Disconnect</button>
</div>

<p id="status">Initializing...</p>

<script>
const ROLE = "__ROLE__";
const CAMERA_ID = "__CAMERA_ID__";
const VIEWER_ID = "__VIEWER_ID__";

const status = document.getElementById("status");

let peer = null;
let localStream = null;
let activeCall = null;

function message(text) {
    status.innerText = text;
}

if (ROLE === "phone") {
    document.getElementById("phone").style.display = "block";
} else {
    document.getElementById("laptop").style.display = "block";
}

function createPeer(id) {
    if (typeof Peer === "undefined") {
        message("PeerJS failed to load. Check internet.");
        return;
    }

    peer = new Peer(id, {
        host: "0.peerjs.com",
        port: 443,
        path: "/",
        secure: true,
        config: {
            iceServers: [
                { urls: "stun:stun.l.google.com:19302" },
                { urls: "stun:stun1.l.google.com:19302" }
            ]
        }
    });

    peer.on("open", () => {
        message("Device registered. Ready to connect.");
    });

    peer.on("error", (err) => {
        message("Connection error: " + err.message);
    });

    peer.on("disconnected", () => {
        message("Signalling disconnected. Reload the page.");
    });
}

if (ROLE === "phone") {
    createPeer(CAMERA_ID);

    document.getElementById("start").onclick = async () => {
        try {
            localStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });

            document.getElementById("localVideo").srcObject = localStream;
            message("Camera started. Open laptop and connect.");
        } catch (error) {
            message("Camera error: " + error.message);
        }
    };

    document.getElementById("stop").onclick = () => {
        if (activeCall) {
            activeCall.close();
            activeCall = null;
        }
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
        }
        document.getElementById("localVideo").srcObject = null;
        message("Camera stopped.");
    };

    function enableCameraCalls() {
        if (!peer) return;
        peer.on("call", call => {
            if (!localStream) {
                call.close();
                message("Start camera before connecting.");
                return;
            }
            activeCall = call;
            call.answer(localStream);
            message("Laptop connected. Streaming live video.");
            call.on("close", () => {
                activeCall = null;
                message("Laptop disconnected.");
            });
        });
    }

    if (peer) {
        peer.on("open", enableCameraCalls);
    }
} else {
    createPeer(VIEWER_ID);

    document.getElementById("connect").onclick = () => {
        if (!peer || peer.destroyed) {
            message("Device not ready. Reload page.");
            return;
        }
        if (activeCall) {
            activeCall.close();
        }
        message("Connecting to phone...");
        activeCall = peer.call(CAMERA_ID, new MediaStream());
        if (!activeCall) {
            message("Could not start connection.");
            return;
        }
        activeCall.on("stream", stream => {
            document.getElementById("remoteVideo").srcObject = stream;
            message("Connected! Live camera streaming.");
        });
        activeCall.on("close", () => {
            document.getElementById("remoteVideo").srcObject = null;
            message("Connection closed.");
        });
        activeCall.on("error", error => {
            message("Call error: " + error.message);
        });
    };

    document.getElementById("disconnect").onclick = () => {
        if (activeCall) {
            activeCall.close();
            activeCall = null;
        }
        document.getElementById("remoteVideo").srcObject = null;
        message("Disconnected.");
    };
}
</script>

</body>
</html>
"""

html = (
    html.replace("__ROLE__", role_js)
    .replace("__CAMERA_ID__", camera_id)
    .replace("__VIEWER_ID__", viewer_id)
)

components.html(
    html,
    height=620,
    scrolling=True
)

st.divider()

st.subheader("How it works")

st.write("""
1. Phone opens the camera.
2. WebRTC establishes a connection.
3. Camera video streams to the laptop.
4. Video is not intentionally recorded or saved by this application.
""")

st.warning(
    "Use a private room code. Keep the camera page open while streaming. "
    "This demonstration uses STUN but does not include a TURN relay, "
    "so some networks may prevent connection."
)
''')

# Download the written file
from google.colab import files
files.download("app.py")
