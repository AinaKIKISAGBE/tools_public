"tools public" 


/* réduir le temps de traitement dans les étapes data */
/* En revanche, lire une table compressée demande plus de CPU donc peut influer sur d'autres traitements. */
option compress=yes;
 
https://www.developpez.net/forums/d775073/logiciels/solutions-d-entreprise/business-intelligence/sas/sas-base/fusion-grosses-tables-sql-optimiser-temps-execution/

utiliser les paramètres bufno bufsize corrects; l'option threads et éventuellement dbsliceparm=(all,8), 8 ou plus selon votre machine

Il y a aussi les buffers oracle en read update ...
Il y a aussi le DBIDIRECTEXEC
Il y a aussi le DBCOMMIT
Il y a aussi le dbslice
Il y a aussi l’option MULTI_DATASRC_OPT positionnée sur le LIBNAME (limite 4500 valeurs), sas génère automatiquement des clauses in

Attention les bulks ne sont pas souvent rentables car sas commence par créer une table dans un format particulier avant qu'Oracle ne charge cette table et de ce fait les perfs globales ne sont plus bonnes.



/* simulate an index on variable: age */ 
data work.class_index; 
 set sashelp.class; 
 row_id=_n_; 
 keep age row_id; 
run; 
proc sort data=work.class_index; 
 by age row_id; 
run; 
data work.class_index; 
 keep age rid; 
 retain age rid; 
 length rid $20; 
 set work.class_index; 
 by age; 
 if first.age then 
 rid = trim(put(row_id,best.-L)); 
 else 
 rid = trim(rid) || ',' || 
 trim(put(row_id,best.-L)); 
 if last.age then output; 
run;

https://support.sas.com/resources/papers/proceedings/proceedings/sugi31/244-31.pdf


/* Create it */ 
 declare hash h_small ();
/* Define his stricture */ 
 length keyvar smallvar1-smallvar4 8; 
 rc = h_small.DefineKey ( "keyvar" ); 
 rc = h_small.DefineData ( "smallvar1","smallvar2","smallvar3","smallvar4"); 
 rc = h_small.DefineDone (); 
/* les rc ont des valeurs retour (0 indique succès) */
/* Fill it */ 
 do until ( eof_small ); 
 	set work.small (keep=keyvar smallvar1-smallvar4) end = eof_small; 
 	rc = h_small.add (); 
 end; 
/* Access it */ 
 do until ( eof_big); 
 	set work.big end = eof_big; 
 	smallvar1=.; smallvar2=.; smallvar3=.; smallvar4=.; 
 	rc = h_small.find (); 
 	output; 
 end;

if rc ne 0 then do; 
	 smallvar1=.; smallvar2=.; smallvar3=.; smallvar4=.; 
 end;




/* merge with memory table (no sorting or indexing required!) */ 
data work.hash_merge (drop=rc i); 
 /* Create it */ 
 declare hash h_small (); 
 /* Define it */ 
 length keyvar smallvar1-smallvar20 8; 
 array smallvar(20); 
 rc = h_small.DefineKey ( “keyvar” ); 
 rc = h_small.DefineData ( “smallvar1”,”smallvar2”,”smallvar3”,”smallvar4”, 
 “smallvar5”,”smallvar6”,”smallvar7”,”smallvar8”, 
 “smallvar9”,”smallvar10”,”smallvar11”,”smallvar12”, 
 “smallvar13”,”smallvar14”,”smallvar15”,”smallvar16”, 
 “smallvar17”,”smallvar18”,”smallvar19”,”smallvar20” ); 
 rc = h_small.DefineDone (); 
 /* Fill it */ 
 do until ( eof_small ); 
 set work.small end = eof_small; 
 rc = h_small.add (); 
 end; 
 /* Merge it */ 
 do until ( eof_large ); 
 set work.large end = eof_large; 
 /* this loop initializes variables before merging from h_small */ 
 do i=lbound(smallvar) to hbound(smallvar); 
 smallvar(i) = .; 
 end; 
 rc = h_small.find (); 
 output; 
 end; 
run; 


/* Fill it */ 
 maxkeyseq=0; 
 do until ( eof_large ); 
 set work.large(keep=keyvar) end = eof_large; 
 by keyvar; 
 if first.keyvar then keyseq=0; 
 keyseq+1; 
 rc = h_large.add (); 
 if last.keyvar then maxkeyseq=max(maxkeyseq,keyseq); 
 end; 
 /* Merge it */ 
 do until ( eof_small ); 
 set work.small end = eof_small; 
 do keyseq=1 to maxkeyseq; 
 do i=lbound(largevar) to hbound(largevar);
 largevar(i) = .; 
 end; 
 rc = h_large.find (); 
 output; 
 end; 
 drop maxkeyseq; 
 end; 


rc = h.output(dataset: "work.out");


https://www.linkedin.com/pulse/sas-hash-tables-patrick-cuba



%let temps_debut = %sysfunc(time());
%let temps_fin = %sysfunc(time());

%let duree = %sysevalf(&temps_fin.-&temps_debut.);
%put Durée d’exécution : &duree.;


 

https://www.developpez.net/forums/d1205758/logiciels/solutions-d-entreprise/business-intelligence/sas/macro/macros-sas-traiter-liste-fichiers-d-meme-repertoire/

data fichiers (keep=fichiers);
length fichiers $256;
	fich=filename('fich',"C:\MonDossier\");
	/* ouverture du répertoire */
	did=dopen('fich');
	/* comptage du nb de fichier */
	nb_fich=dnum(did);
	do i=1 TO nb_fich;
		fichiers=dread(did,i);
		output;
	end;
	/* fermeture du répertoire */
	rc=dclose(did);
run;


https://teaching.slmc.fr/sas_macro/handout.pdf
%MACRO test(deb,fin);
DATA chomage;
SET
%DO %WHILE (&deb < &fin);
chomage&deb
%LET deb = %EVAL(&deb +
1);
%END;
;
RUN;
%MEND;

https://support.sas.com/resources/papers/proceedings11/113-2011.pdf
MACRO DO_LIST;
%DO I = 1 %TO 50;
 COL&I._LINE1
%END;
%MEND DO_LIST;

https://cmie.padoa.fr/employee/login?email=aina.kikisagbe@mel.lincoln.fr&token=FJGCFF



