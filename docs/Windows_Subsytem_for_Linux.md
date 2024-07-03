# Guide to use the Tiny ML Modelmaker on a Windows PC

## Step 1: Install WSL on PC

* Firstly, you need to have administrator permissions on your PC

* Open the Windows PowerShell (or Command Prompt), right click and choose "Run as Administrator"

* To install WSL, you can do the following:
```commandline
 wsl --install --web-download --distribution Ubuntu-22.04  
```
You will have to restart the system after this installation completes 

#### Note: The above command has multiple internal steps. Sometimes it may error after a few steps. You may have to run the same command multiple times, no harm in it.
#### Official documentation: [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/about)

## Step 2: Create a local login

* After installation succeeds, and reboot is done,

* Reopen PowerShell and run ``` wsl --install --web-download --distribution Ubuntu-22.04  ``` command once again, this time it invokes wsl and you should see something like this:

* You can enter a username and password of your choice. Has no relation with your enterprise login/PC actual name
```commandline
PS C:\WINDOWS\system32> wsl --install --distribution Ubuntu-22.04 --web-download
Ubuntu 22.04 LTS is already installed.
Launching Ubuntu 22.04 LTS...                                                                                           
Installing, this may take a few minutes...
Please create a default UNIX user account. The username does not need to match your Windows username.
For more information visit: https://aka.ms/wslusers
Enter new UNIX username: user_adithya
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
/home/user_adithya/.hushlogin file.
user_adithya@computer:~$  
```
## Important Notes:
* How do I access files in WSL2 from Windows?
  * On the File Explorer, type: ```\\wsl.localhost\Ubuntu-22.04``` and hit Enter
* How do I access files in my Windows machine from WSL2?
  * If you are in Ubuntu and need access to a file on a Windows drive (e.g. C:), then you'll find those are (by default) auto-mounted for you:
  * ```ls /mnt/c/ ```
  * There are some nuances in working with files on a Windows drive from within WSL, especially around permissions and performance. 
  * You'll typically want to keep any project files inside the Ubuntu ext4 filesystem (e.g. under your /home/user_adithya directory).
  * But you can certainly access, copy, and move files around between the drives as needed.

## Step 3: Keeping your WSL system upto date
### Step 3.1 (Optional)

* The subsequent steps require WSL to connect to the internet.
* Depending on your firewall/VPN/proxy settings, you may face different set of errors when trying to access the internet
* The following two changes may seem generally relevant:

#### 3.1.1: Changes to /etc/resolv.conf
Type on the terminal `sudo nano /etc/resolv.conf` and put the following contents on the file:
```bash
nameserver 8.8.8.8  # Required to search google
```
* Note: For TI, I had to do the following changes to `/etc/resolv.conf`
```bash
search dhcp.ti.com ent.ti.com itg.ti.com ext.ti.com  # Required to connect to internal servers
nameserver 192.0.2.2
nameserver 192.0.2.3
nameserver 8.8.8.8
```
#### 3.1.2: Changes to /etc/wsl.conf
Type on the terminal `sudo nano /etc/wsl.conf` and put the following contents on the file:
```bash
[network]
generateResolvConf = false
```
#### 3.1.3: Making `/etc/resolv.conf` immutable
```bash
sudo chattr +i /etc/resolv.conf
```

#### 3.1.4: Internal Proxy settings in `~/.bashrc`
This may be user dependant, for example, at TI, the proxies had to be set in `~/.bashrc` to access the internet



### Step 3.2
#### 3.2.1: System level proxy for `apt` to work

#### Note: The following is for TI network connected machines only, it is meant to serve as an example to how proxy needs to be set for `apt` to work in WSL environment

To use apt, you will need to add the TI proxy.

1. Add the following to `/etc/apt/apt.conf` (even if the file doesn't exist) with nano or vi (with sudo)
```bash
Acquire::http::proxy "http://webproxy.ext.ti.com:80";
Acquire::https::proxy "http://webproxy.ext.ti.com:80";
```

2. Create the following `/etc/profile.d/proxy.sh` file (with sudo)

```bash
export http_proxy="http://webproxy.ext.ti.com:80/"
export https_proxy="http://webproxy.ext.ti.com:80/"
export no_proxy="localhost,127.0.0.1,.ti.com"
export HTTP_PROXY="http://webproxy.ext.ti.com:80/"
export HTTPS_PROXY="http://webproxy.ext.ti.com:80/"
export NO_PROXY="localhost,127.0.0.1,.ti.com"
```

3. Shutdown WSL. In a separate command prompt, type the following
```bash 
wsl --shutdown
```
4. Reopen WSL2
5. Verify by running the env command
```bash
env
```

#### 3.2.2 WSL Ubuntu system may not be upto date, hence the following commands will help you get it upto date
```commandline
sudo apt update
sudo apt upgrade
```

* Also, to install Python later, we need an acceptable C compiler to be installed in the WSL. To install gcc,g++,make etc,
```commandline
sudo apt-get install build-essential
```

## Step 4: Set up Visual Studio Code (Optional)
* On a WSL, you can't use any GUI applications. However, for development, you could install Visual Studio Code on your Windows PC and connect to your WSL.
* Refer [this article](https://learn.microsoft.com/en-us/windows/python/web-frameworks#set-up-visual-studio-code) from Microsoft 

## Step 5: Installing Tiny ML Modelmaker
- Follow the same steps as in the main [documentation](../README.md)