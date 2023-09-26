import sys
import hmac
from binascii import a2b_hex, b2a_hex
from hashlib import pbkdf2_hmac, sha1, md5


class Crypto_tool:
    def __init__(self, pwd_list, ssid, NonceA, NonceS, apMAC, cliMAC, data1, data2, data3, target_mic1, target_mic2, target_mic3) -> None:
        self.__pwd_list__   = pwd_list
        self.__ssid__       = ssid
        self.__NonceA__     = NonceA
        self.__NonceS__     = NonceS
        self.__apMAC__      = apMAC
        self.__cliMAC__     = cliMAC
        self.__data1__      = data1
        self.__data2__      = data2
        self.__data3__      = data3
        self.__target_mic1__= target_mic1
        self.__target_mic2__= target_mic2
        self.__target_mic3__= target_mic3
        

    def PRFn(self, K, A, B, n = 512):
        i = 0
        R = b''
        while len(R)*8 < n:
            data = A + b'\x00' + B + bytes([i])
            r = hmac.new(K, data, sha1).digest()
            R += r
            i += 1
        return R[:n//8]

    def FindAB(self):
        A = b"Pairwise key expansion"
        B = min(self.__apMAC__, self.__cliMAC__) + max(self.__apMAC__, self.__cliMAC__) + min(self.__NonceA__, self.__NonceS__) + max(self.__NonceA__, self.__NonceS__)
        return (A, B)
    
    #   Password is got from pwd_list
    def Generate_mic(self, pwd, A, B, datas, hmac_type):
        PMK = pbkdf2_hmac('sha1', pwd.encode('ascii'), self.__ssid__ .encode('ascii'), 4096, 32)
        PTK = self.PRFn(PMK, A, B)
        computed_mics = [hmac.new(PTK[0:16], data, hmac_type).digest() for data in datas]

        return computed_mics, PMK, PTK


    def solve(self) -> None:
        A, B = self.FindAB()
        for pwd in self.__pwd_list__:
            
            computed_mics, PMK, PTK = self.Generate_mic(pwd, A, B, [self.__data1__, self.__data2__, self.__data3__], md5)
            computed_mic1 = b2a_hex(computed_mics[0]).decode()
            if computed_mic1 != self.__target_mic1__:
                #print("Wrong in mic 1")
                continue
            #print("MIC 1 MATCHED")
            computed_mic2 = b2a_hex(computed_mics[1]).decode()
            if computed_mic2 != self.__target_mic2__:
                #print("Wrong in mic 2")
                continue
            #print("MIC 2 MATCHED")
            computed_mic3 = b2a_hex(computed_mics[2]).decode()
            if computed_mic3 != self.__target_mic3__:
                #print("Wrong in mic 3")
                continue
            #print("MIC 3 MATCHED")
            
            # Print PMK
            print("PMK:\t\t" + b2a_hex(PMK).decode().upper() + '\n')

            # Print PTK
            print("PTK:\t\t" + b2a_hex(PTK).decode().upper() + '\n')
            
            # compare MIC result
            print('First target MIC:\t\t' + self.__target_mic1__)
            print('First computed MIC1:\t\t' + computed_mic1)
            print('\nSecond target MIC2:\t\t' + self.__target_mic2__)
            print('Second computed MIC2:\t\t' + computed_mic2)
            print('\nThird target MIC3:\t\t' + self.__target_mic3__)
            print('Third computed MIC3:\t\t' + computed_mic3)
            
            # Show password
            print('\n\nPassword found:\t\t\t' + pwd)

            return pwd
        print("Can't find the correct password!")
        return None



   
    
def main() -> int:
    with open('pwd_list.txt') as f:
        pwd_list = []
        for l in f:
            pwd_list.append("aaaa" + l.strip())
    #pwd_list = ["aaaababa"]
    #ssid name
    ssid = "M1WPA"
    #ANonce
    aNonce = a2b_hex('7c67f224a6e08193230feeb0eff9a07ec6cbf0163f962ba34d31dbdb2bc69d8d')
    # 1b34324a9344e5c86cdd185ff97dab18d1eb7228573eaa045b122219c9f0fe57
    #SNonce
    sNonce = a2b_hex("eea4124e3facf8e0270db587fceee4da1c2a689be96b931fc26d35b4c7dbbbae")
    
    #Authenticator MAC (AP)
    apMac = a2b_hex("000e3569fed5")
    
    #Station address: MAC of client
    cliMac = a2b_hex("76d9ac111684")
    
    #The first target MIC
    mic1 = "082793ece524d399179cbc039e0239e4"

    #The entire 802.1x frame of the second handshake message
    data1 = "01030079fe010900200000000000000002eea4124e3facf8e0270db587fceee4da1c2a689be96b931fc26d35b4c7dbbbae0000000000000000000000000000000000000000000000000000000000000000082793ece524d399179cbc039e0239e4001add180050f20101000050f20201000050f20201000050f2022a00"
    #Set MIC field to zero
    data1 = a2b_hex(data1.replace(mic1, "0"*32))

    #The second target MIC
    mic2 = "e2180d61d789a81d422382819e3efe4e"

    #The entire 802.1x frame of the third handshake message 
    data2 = "01030077fe01c9002000000000000000037c67f224a6e08193230feeb0eff9a07ec6cbf0163f962ba34d31dbdb2bc69d8d0000000000000000000000000000000000000000000000000000000000000000e2180d61d789a81d422382819e3efe4e0018dd160050f20101000050f20201000050f20201000050f202"
    #Set MIC field to zero
    data2 = a2b_hex(data2.replace(mic2, "0"*32))

    #The third target MIC
    mic3 = "adda25ccf2fcaecfd18b37f2b2ffafd2"
    #The entire 802.1x frame of the forth handshake message
    data3 = "0103005ffe01090020000000000000000300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000adda25ccf2fcaecfd18b37f2b2ffafd20000"
    #Set MIC field to zero
    data3 = a2b_hex(data3.replace(mic3, "0"*32))

    crypto = Crypto_tool(pwd_list, ssid, aNonce, sNonce, apMac, cliMac, data1, data2, data3, mic1, mic2, mic3)
    crypto.solve()
    

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit