from magic import MagicWorker

if __name__ == '__main__':
    magic = MagicWorker()
    ans = magic.predict('Не знаю, как подлкючиться к интернету')
    print(ans)