import subprocess
import numpy as np 


program_list = ["Minors.cpp"]


str_ = input("\nDo you need the approximate computation? (yes/no): ")

if str_ == 'yes': 

    for program in program_list:
        
        if program == "Minors.cpp":
            
            cmd = "Minors.cpp"
            subprocess.call(["g++", cmd])
            subprocess.call("./a.out")
            print("\nFinished: " + program)
                
        else:
            cmd = program
            subprocess.call(["g++", cmd])
            subprocess.call("./a.out")
            print("\nFinished: " + program)
                
       