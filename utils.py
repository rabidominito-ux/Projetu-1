def calcular_resultado(lista_notas):
    # Ezemplu kálkulu simples
    media = sum(lista_notas) / len(lista_notas)
    if media >= 80:
        return "Excelente"
    elif media >= 60:
        return "Bom"
    else:
        return "Satisfatório"
