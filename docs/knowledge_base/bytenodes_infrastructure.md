# Bytenodes Infrastructure & Architecture Guide

## Overview
Bytenodes is a high-performance cloud virtualization and gaming hosting provider. The infrastructure consists of dedicated physical bare-metal Proxmox VE hosts and geographically distributed cluster nodes.

## Virtualization Topology
- **Host Physical Proxmox:** Primary node located on IP `208.84.103.84`, managing virtualization containers (LXC) and full hardware-accelerated KVM virtual machines.
- **Node Kazuha:** Dedicated node handling intensive game server virtualization.
- **Node Xiao (`192.168.8.20`):** High-throughput microservice compute node running Docker daemons, application gateways, and containerized game instances.
- **Node Anaxa & Furina:** Specialized redundancy nodes providing high availability storage and failover routing.

## Pterodactyl Panel Infrastructure
- **Panel Location:** Hosted on internal VM `192.168.8.10` reachable publicly via `panel.bytenodes.id`.
- **Services:** Runs Nginx reverse proxy, PHP-FPM 8.2, Redis cache, and MySQL database daemon.
- **Wings Daemons:** Wings communicating via encrypted gRPC on port 8080 across all compute nodes.

## Backup Gateway & Disaster Recovery
- Automated snapshots are captured daily at 03:00 WIB.
- Encrypted volumes are streamed via S3 Backup Gateway to Wasabi and Cloudflare R2 object storage.
