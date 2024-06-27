# Guide to use the Tiny ML Modelmaker on a Windows PC

## Step 1: Install WSL on PC

Firstly, you need to have administrator permissions on your PC

Open the Windows PowerShell (or Command Prompt), right click and choose "Run as Administrator"

To install WSL, you can do the following:
```bash
 wsl --install --web-download --distribution Ubuntu-22.04  
```
You will have to restart the system after this installation completes 

#### Note: The above command has multiple internal steps. Sometimes it may error after a few steps. You may have to run the same command multiple times, no harm in it.
#### Official documentation: [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/about)

## Step 2: Create a local login

After installation succeeds, and reboot is done,

Reopen PowerShell and run ``` wsl --install --web-download --distribution Ubuntu-22.04  ``` command once again, this time it invokes wsl and you should see something like this:

You can enter a username and password of your choice. Has no relation with your enterprise login/PC actual name
```bash
PS C:\WINDOWS\system32> wsl --install --distribution Ubuntu-22.04 --web-download
Ubuntu 22.04 LTS is already installed.
Launching Ubuntu 22.04 LTS...                                                                                           
Installing, this may take a few minutes...
Please create a default UNIX user account. The username does not need to match your Windows username.
For more information visit: https://aka.ms/wslusers
Enter new UNIX username: username123
New password:
Retype new password:
passwd: password updated successfully
The operation completed successfully.
Installation successful!
To run a command as administrator (user "root"), use "sudo <command>".
See "man sudo_root" for details.
Welcome to Ubuntu 22.04.2 LTS (GNU/Linux 5.15.153.1-microsoft-standard-WSL2 x86_64)
* Documentation:  https://help.ubuntu.com
* Management:     https://landscape.canonical.com
* Support:        https://ubuntu.com/advantage
This message is shown once a day. To disable it please create the
/home/username123/.hushlogin file.
username123@computer:~$  
```
## Important Notes:
* How do I access files in WSL2 from Windows?
  * On the File Explorer, type: ```\\wsl.localhost\Ubuntu-22.04``` and hit Enter
* How do I access files in my Windows machine from WSL2?
  * If you are in Ubuntu and need access to a file on a Windows drive (e.g. C:), then you'll find those are (by default) auto-mounted for you:
  * ```ls /mnt/c/ ```
  * There are some nuances in working with files on a Windows drive from within WSL, especially around permissions and performance. 
  * You'll typically want to keep any project files inside the Ubuntu ext4 filesystem (e.g. under your /home/username123 directory).
  * But you can certainly access, copy, and move files around between the drives as needed.

## Step 3: Installing Tiny ML Modelmaker
- Follow the same steps as in the main [documentation](../README.md)