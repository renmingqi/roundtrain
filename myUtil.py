import paramiko
import logging
import  os
##define log level
logging.basicConfig(level=logging.INFO)

##params for SSH connection, at Azure forftp@12.78.188.19


sub1 = '.index'
sub2 = '.meta'
sub3 = '.data-00000-of-00001'

## Model Path and remote path.

local_path = "/tmp/ts/model.ckpt"

remote_path = "/tmp/model.ckpt"
local_name1=local_path + sub1
local_name2=local_path + sub2
local_name3=local_path + sub3
local_names = [local_name1,local_name2,local_name3]


remote_name1=remote_path + sub1
remote_name2=remote_path + sub2
remote_name3=remote_path + sub3
remote_names = [remote_name1,remote_name2,remote_name3]

sublock=".mmlock"
lock_name1 = remote_name1 +"_s"+ sublock
lock_name2 = remote_name2 +"_s"+ sublock
lock_name3 = remote_name3 +"_s"+ sublock
lock_names = [lock_name1,lock_name2,lock_name3]

def isFilesCreated():
    res = True
    for name in local_names:
        if not os.path.isfile(name):
            res = False
    return res

def getSSHconnection(ipaddr = '13.78.188.19', username = 'forftp',pwd ='uiop[]\\7890-'):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(ipaddr, username=username)
        logging.info("conneted to "+ "@" + username + ipaddr)
    except paramiko.SSHException:
        print("Connection Failed")
        quit()
    return ssh

def commandModifyTime(path):
    res =  "stat  " + path + "  |grep " +  "'Modify'"
    print(res)
    return res

def queryLastModifiedTime(ssh,path):
    stdin, stdout, stderr = ssh.exec_command(commandModifyTime(path))
    res=stdout.read().decode()
    return res

def queryLastModifiedTime3(ssh,paths):
    res = []
    for path in paths:
        stdin, stdout, stderr = ssh.exec_command(commandModifyTime(path))
        res.append(stdout.read().decode())
    return res

def commandTestFile( path):
    res =  "[ -f %s ] && echo \"1\" || echo \"0\" " % path
    print(res)
    return res

def queryFileExistence(ssh,path):
    print(path)
    stdin, stdout, stderr = ssh.exec_command(commandTestFile(path))
    res = int(stdout.read().decode())
    return (res==1)

def queryFileExistence3(ssh,paths):
    res = []
    for path in paths:

        res.append(queryFileExistence(ssh,path))
    return res

def commandCreateLock(lockname):
    res =  "touch %s "% lockname
    print(res)
    return res

def commandRemoveLock(lockname):
    res = "rm %s "% lockname
    print(res)
    return res

def createLock(ssh,lockname):
    ssh.exec_command(commandCreateLock(lockname))

def removeLock(ssh,lockname):

    ssh.exec_command(commandRemoveLock(lockname))


def createLock3(ssh,lock_names):

    for ln in lock_names:

        ssh.exec_command(commandCreateLock(ln))



def removeLock3(ssh,remote_names):
    for rn in remote_names:
        ssh.exec_command(commandCreateLock(rn))

# def createLock(ssh,workerToLock):

def uploadFileCallback(a,b,mtime_last,i,ssh):
    if a==b:
        print("callback_update modified time at "+ remote_names[i])
        mtime_last[i] = queryLastModifiedTime(ssh,remote_names[i])
        print("callback_remove lock at " + lock_names[i])
        removeLock(ssh,lock_names[i])

def uploadFile(mtime_last,sftp,ssh,remote_names,local_names):
    for i in range(0,3):
        ##Once uploaded a file, remove its lock = filename+.lock

        sftp.put(local_names[i],remote_names[i],lambda a,b:uploadFileCallback(a,b,mtime_last,i,ssh))

def downloadFile(sftp,remote_names,local_names):
    for i in [0,1,2]:
        sftp.get(remote_names[i],local_names[i])

def array_equal(aa,ab):
    res =False
    if aa[0]==ab[0]:
        if aa[1]==ab[1]:
            if aa[2]==ab[2]:
                res =True
    return res