import sys

if len(sys.argv) > 1:
    for argument in sys.argv[1:]:
        print(argument)
else:
    print("usage is: python <script name>.py argument")
    quit()