import asyncio
import logging
import os
from typing import Dict, Optional
import json

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer

logger = logging.getLogger(__name__)

class WebRTCHandler:
    """Handles WebRTC peer connections and media streams"""
    
    def __init__(self):
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        self.ice_servers = [
            RTCIceServer(urls=["stun:stun.l.google.com:19302"])
        ]
        self.config = RTCConfiguration(iceServers=self.ice_servers)
    
    async def handle_offer(self, client_id: str, offer: dict) -> dict:
        """Handle WebRTC offer and create answer"""
        try:
            # Create peer connection
            pc = RTCPeerConnection(configuration=self.config)
            self.peer_connections[client_id] = pc
            
            # Set up event handlers
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                logger.info(f"Connection state for {client_id}: {pc.connectionState}")
                if pc.connectionState == "closed":
                    await self.cleanup_connection(client_id)
            
            @pc.on("track")
            async def on_track(track):
                logger.info(f"Received track: {track.kind}")
                if track.kind == "video":
                    # Handle incoming video track
                    await self.handle_video_track(client_id, track)
            
            # Set remote description
            await pc.setRemoteDescription(RTCSessionDescription(
                sdp=offer["sdp"],
                type=offer["type"]
            ))
            
            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }
            
        except Exception as e:
            logger.error(f"Error handling offer: {e}")
            raise
    
    async def handle_answer(self, client_id: str, answer: dict):
        """Handle WebRTC answer"""
        try:
            pc = self.peer_connections.get(client_id)
            if pc:
                await pc.setRemoteDescription(RTCSessionDescription(
                    sdp=answer["sdp"],
                    type=answer["type"]
                ))
        except Exception as e:
            logger.error(f"Error handling answer: {e}")
    
    async def handle_ice_candidate(self, client_id: str, candidate: dict):
        """Handle ICE candidate"""
        try:
            pc = self.peer_connections.get(client_id)
            if pc and candidate:
                await pc.addIceCandidate(candidate)
        except Exception as e:
            logger.error(f"Error handling ICE candidate: {e}")
    
    async def handle_video_track(self, client_id: str, track):
        """Handle incoming video track from client"""
        try:
            logger.info(f"Handling video track for client {client_id}")
            
            # This is where we would process the video frames
            # For now, we'll just log that we received the track
            async for frame in track:
                # Frame processing would happen here
                # For the WebSocket approach, frames are sent separately
                pass
                
        except Exception as e:
            logger.error(f"Error handling video track: {e}")
    
    async def cleanup_connection(self, client_id: str):
        """Clean up peer connection resources"""
        try:
            if client_id in self.peer_connections:
                pc = self.peer_connections[client_id]
                await pc.close()
                del self.peer_connections[client_id]
                logger.info(f"Cleaned up connection for {client_id}")
        except Exception as e:
            logger.error(f"Error cleaning up connection: {e}")
    
    async def get_connection_stats(self, client_id: str) -> Optional[dict]:
        """Get connection statistics"""
        try:
            pc = self.peer_connections.get(client_id)
            if pc:
                stats = await pc.getStats()
                return stats
            return None
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return None
