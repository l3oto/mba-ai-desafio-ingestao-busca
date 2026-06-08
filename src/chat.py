from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado. Digite 'sair' para encerrar.\n")

    while True:
        question = input("PERGUNTA: ").strip()
        if question.lower() in ("sair", "exit", "quit"):
            break
        if not question:
            continue
        answer = chain(question)
        print(f"RESPOSTA: {answer}\n")


if __name__ == "__main__":
    main()