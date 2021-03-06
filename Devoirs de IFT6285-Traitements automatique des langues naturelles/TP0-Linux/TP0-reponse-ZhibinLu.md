# IFT6285 TP0 Reponses

> Zhibin Lu

> 18 janv. 2018

## 1) découper un texte en mots

```
iconv -f Windows-1252 -t UTF-8 zola1.txt > zola1.utf8.txt

cat zola1.txt   | awk 'BEGIN {ok=0} \
                      /DEBUT DU FICHIER/ {ok = 1} \
                      /FIN DU FICHIER/ {ok=0} \
                      {if (ok>10) print $0; else if (ok) ok++}'
```

## 2) calculer une table de fréquence
* le parametre s dans sed: remplace, mais dans tr:Éliminer la redondance.

#### Premiere étape: Gardez les ponctuations mais les séparer
```
./scorie.sh zola1.utf8.txt |sed -e "s/'/' /g" -e "s/\./ \. /g" -e "s/\?/ \? /g" -e "s/\!/ \! /g" -e "s/,/ , /g" -e "s/;/ ; /g" -e "s/-/ - /g" -e "s/;/ ; /g" -e "s/\"/ \" /g" -e "s/:/ : /g" |tr 'A-Z' 'a-z' |tr -d '\r' |tr -s '\n' |tr -s ' ' >zola-pretraite.txt
```
si on veut éliminer les ponctuations:
```
./scorie.sh zola1.utf8.txt |sed -e "s/'/ /g" -e "s/\./ /g" -e "s/\?/ /g" -e "s/\!/ /g" -e "s/,/ /g" -e "s/;/ /g" -e "s/-/ /g" -e "s/;/ /g" -e "s/\"/ /g" -e "s/:/ /g" |tr 'A-Z' 'a-z' |tr -d '\r' |tr -s '\n' |tr -s ' ' >zola-pretraite.txt
ou:
./scorie.sh zola1.utf8.txt | ./tokenizer.sh >zola-pretraite.txt
```

#### Deuxieme étape:
```
cat zola-pretraite.txt |tr ' ' '\n'|sort|uniq -c|sort -k1,1nr|less
```

## 3) lister tous les mots d'un texte qui ...
1. ont exactement 4 caractères
```
cat zola-pretraite.txt |tr ' ' '\n'|grep -iE "^....$" |sort|uniq -c|sort -k 1,1nr|more
```
2. commencent par le préfixe p
```
cat zola-pretraite.txt |tr ' ' '\n'|grep -iE "^abom" |sort|uniq -c|sort -k 1,1nr|less
```
3. terminent par le suffixe s
```
cat zola-pretraite.txt |tr ' ' '\n'|grep -iE "lissait$" |sort|uniq -c|sort -k 1,1nr|more
```
4. contiennent au milieu (ni au début, ni à la fin) la chaîne a
```
cat zola-pretraite.txt |tr ' ' '\n'|grep -iE '[^g]glou[^u]' |sort|uniq -c|sort -k 1,1nr|more
```
5. sont vus au moins n fois dans le texte
```
cat zola-pretraite.txt |tr ' ' '\n'|sort|uniq -c|sort -k1,1nr|awk '$1>2000 {print $0}'
```
6. exactement n fois dans le texte
```
cat zola-pretraite.txt |tr ' ' '\n'|sort|uniq -c|sort -k1,1nr|awk '$1==108 {print $0}'
```
7. lus à l'envers sont encore des mots
```
cat zola-pretraite.txt |tr -s ' '|tr ' ' '\n'|sort|uniq |tr '\n' ' '|awk '{
  lens=split($0,array," ");
  for (i=1;i<=lens;i++){
    if (length(array[i])>3){
      cmd="echo \"" array[i] "\" | rev";
      cmd |getline envstr;
      for (j=1;j<=lens;j++){
        if (array[j]==envstr){
          print array[i] " " envstr "\n";
          break;
        }
      }
      close(cmd);
    }
  }
}'
```

## 4) respirer
Passe

## 5) calculer les bigrammes et leur fréquence
```
cat zola-pretraite.txt  |tr '\n' ' ' |tr -s ' ' |awk '{
  lens=split($0,array," ");
  for (i=1;i<lens;i++){
    print array[i] " " array[i+1];
  }
}' |sort|uniq -c|sort -k1,1nr|more
```

## 6) Écrire un programme capable de lister l'ensemble des trigrammes d'un texte et d'afficher leur fréquence d'occurrence
```
cat zola-pretraite.txt  |tr '\n' ' ' |tr -s ' ' |awk '{
  lens=split($0,array," ");
  for (i=1;i<lens-1;i++){
    print array[i] " " array[i+1] " " array[i+2];
  }
}' |sort|uniq -c|sort -k1,1nr|more
```

## 7) afficher les mots qui suivent un mot donné
```
cat zola-pretraite.txt  |tr '\n' ' ' |tr -s ' ' |awk '{
  lens=split($0,array," ");
  for (i=1;i<lens;i++){
    print array[i] " " array[i+1];
  }
}' |sort|uniq -c|sort -k1,1nr| grep -iE " le " |more
```

## 8) afficher les mots qui suivent un bigramme donné
```
cat zola-pretraite.txt  |tr '\n' ' ' |tr -s ' ' |awk '{
  lens=split($0,array," ");
  for (i=1;i<lens-1;i++){
    print array[i] " " array[i+1] " " array[i+2];
  }
}' |sort|uniq -c|sort -k1,1nr | grep -iE " la belle " |more
```

## 9) afficher les bigrammes et trigrammes caractéristiques d'un texte (un mot assez long)
```
cat zola-pretraite.txt  |tr '\n' ' ' |tr -s ' ' |awk '{
  lens=split($0,array," ");
  for (i=1;i<lens-1;i++){
    if (length(array[i])>6 || length(array[i+1])>6 || length(array[i+2])>6){
      print array[i] " " array[i+1] " " array[i+2];
    }
  }
}' |sort|uniq -c|sort -k1,1nr|more
```

## 10) récupérer une liste de mots outils du français
```
iconv -f ISO-8859-1 -t UTF-8 stop.txt > stop.utf8.txt

cat stop.utf8.txt |sed 's/\ \{1,\}\|.*//g' |sed  '/^$/d' | wc -w
```

## 11) retirer les mots outils dans un texte
```
cat zola-pretraite.txt  |tr '\n' ' ' |tr -s ' ' |awk 'BEGIN{
  cmd="cat french.stop.louis.txt | xargs echo -n";
  cmd |getline stopstr;
  close(cmd);
  split(stopstr,stops," ");
}
{
  lens=split($0,array," ");
  for (i=1;i<=length(array);i++){
    for (j=1;j<=length(stops);j++){
      if (array[i]==stops[j]) {
        array[i]="STOP";
      }
    }
  }
}
END{for (i=1;i<=length(array);i++){print array[i];}}' |more
```

## 12) retrouver les mots manquants
1. Remplacez chaque mot GUESS par le mot le plus fréquent.
```
cat zola1.guess.utf8.txt  |tr '\n' ' ' |tr -s ' ' |awk '
{
  lens=split($0,array," ");
  for (i=1;i<=length(array);i++){
    if (array[i]=="GUESS"){
      array[i]="de";
    }
    print array[i];
  }
}'  | sed '/^$/d' >zola1.cand.louis.txt

./evaluation.sh zola1.guess.utf8.txt zola1.cand.louis.txt zola1.toks.utf8.txt
#Resultat: good: 301 bad: 6990 err: 95,87
```

2. Remplacez chaque mot GUESS par le mot qui a le plus de chance de suivre le mot qui précède le mot GUESS
```
cat zola-pretraite.txt  |tr '\n' ' ' |tr -s ' ' |awk '{
  lens=split($0,array," ");
  for (i=1;i<lens;i++){
    print array[i] " " array[i+1];
  }
}' |sort|uniq -c|sort -k1,1nr|awk '{print $2" "$3","}' > zola1.bigrammes2.louis.txt

cat zola1.guess.utf8.txt  |tr '\n' ' ' |tr -s ' ' |awk 'BEGIN{
  cmd="cat zola1.bigrammes2.louis.txt | xargs echo -n";
  cmd |getline bigrammes;
  close(cmd);
  split(bigrammes,stops,", ");
}
{
  lens=split($0,array," ");
  print array[1];
  for (i=2;i<=length(array);i++){
    if (array[i]=="GUESS"){
      for (j=1;j<=length(stops);j++){
        split(stops[j],onebigramme," ");
        if (array[i-1]==onebigramme[1]){
          array[i]=onebigramme[2];
          break;
        }
      }
    }
    print array[i];
  }
}' | sed '/^$/d' >zola1.cand.louis2.txt

./evaluation.sh zola1.guess.utf8.txt zola1.cand.louis2.txt zola1.toks.utf8.txt

#Resultat: good: 1377 bad: 5914 err: 81,11
```