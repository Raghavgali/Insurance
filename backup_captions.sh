#!/bin/bash
mkdir -p /home/ubuntu/Insurance/logs
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
/usr/bin/rclone copy \
    --transfers=16 \
    --config /home/ubuntu/.config/rclone/rclone.conf \
    --progress \
    /home/ubuntu/Insurance/data/captions \
    gdrive:"Insurance/captions_backup/$timestamp" && \
echo "Backup completed at $timestamp"
echo "[CRON] Backup completed at $timestamp" >> /home/ubuntu/Insurance/logs/backup.log