# infrastructure/ansible

Ansible playbooks for Ubuntu VPS server configuration.

Planned playbooks:
- `setup.yml`     — base server setup (nginx, node, python, pm2)
- `deploy.yml`    — rolling deployment of Trainer API
- `backup.yml`    — configure cron backup jobs
- `hardening.yml` — SSH hardening, firewall, fail2ban
