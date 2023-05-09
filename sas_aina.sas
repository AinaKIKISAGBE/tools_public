
/* sequential access */ 
data work.sequential;
 set sashelp.class;
 put _all_;
 output;
run;


/* selectionner les ligne ayant pour id 2,3 et 9*/
/* direct access */
data work.direct;
 do i=2, 3, 9;
	 set sashelp.class point=i;
	 put _all_;
	 output;
 end;
 stop;
run; 


/* implicit looping */ 
/*data work.sequential;  
 	set sashelp.class; 
 	put _all_; 
 	output; 
run; 
*/

/* explicit looping */
data work.sequential3;
	do until (eof);
		set sashelp.class end=eof;
		put _all_;
		output;
 	end;
run;



/* simulate an index on variable: age */
data work.class_index1;
 set sashelp.class;
 row_id=_n_;
 keep age row_id;
run;
proc sort data=work.class_index1;
 by age row_id;
run;

/*lister tous les id regrouper par la cl� age */
data work.class_index;
	keep age rid;
 	retain age rid;
 	length rid $20;
 	set work.class_index1;
 		by age;
 		if first.age then rid = trim(put(row_id,best.-L));
 		else rid = trim(rid) || ',' || trim(put(row_id,best.-L));
 		if last.age then output;
run; 


/* trier directement par age ordre decroissant par d�faut*/
options msglevel=i;
/* class with index*/
data work.class (index=(age));
 set sashelp.class;
run;


/* sequential access in ascending order (ordre croissant)*/
data work.sortedclass;
 set work.class;
 by age;
run; 


/* direct access filtrer sur age = 13 ou 14*/
data work.direct;
 	do age=13,14;
 		do until (eof);
 			set class key=age end=eof;
			/* _IORC_=0 si une correspondance a �t� trouv�e */
 			if _IORC_=0 then do; /* 0 indicates a match was found */ 
 				put _all_; /* envoyer les correspondances trouv�es � key (age = 13 ou age=14) dans class vers direct */
 				output;
 			end;
			/*s'il n'y a pas de correspondance, r�initialisez le drapeau d'erreur et continuez*/
 			else _ERROR_=0; /* if no match, reset the error flag and continue */
 		end;
 	end;
 	stop;
run; 


/* direct access */
data work.driver;
 	age=13; output;
 	age=14; output;
run;

data work.direct;
 	set work.driver; /* <- sequential access & implicit loop */
 		do until (eof); /* <- explicit loop */
 			set work.class key=age end=eof; /* <- direct access */
 			if _IORC_=0 then do; /* 0 indicates a match was found */
 				put _all_;
 				output;
 			end;
 			else _ERROR_=0; /* if no match, reset the error flag and continue */
 		end;
run;
 






%macro testest(nb_obs_set);
/* clear log */
dm 'log;clear;';
%let large_obs = /*50000*/ &nb_obs_set.;
data work.small ( keep = keyvar small: )
    work.large ( keep = keyvar large: );
    array keys(1:&nb_obs_set.) $1 _temporary_;
 	length keyvar 8;
 	array smallvar [20]; retain smallvar 12;
 	array largevar [682]; retain largevar 55;
 	do _i_ = 1 to &large_obs ;
 		keyvar = ceil (ranuni(1) * &large_obs);
 		if keys(keyvar) = ' ' then do;
 			output large;
 			if ranuni(1) < 1/5 then output small;
 			keys(keyvar) = 'X';
 		end;
 	end;
run; 

proc sql noprint;
    select count(*) into :nbligne_small from small;
quit;
proc sql noprint;
    select count(*) into :nbligne_large from large;
quit;
%put nbligne_small = &nbligne_small.;
%put nbligne_large = &nbligne_large.;


data small (keep= smallvar smallvar1 smallvar2); 
	set small; 
	/*if _n_ <= 8;*/ 
	smallvar = keyvar;
run;

data large (keep= smallvar largevar1 largevar2); 
	set large; 
/*	if _n_ <= 8; */
	smallvar = keyvar;
run;

data small_sauv; set small; run;
data large_sauv; set large; run;


/* 4- jointure sql */
/* basic match-merge with sort */
/* clear log */
dm 'log;clear;';
data small; set small_sauv; run;
data large; set large_sauv; run;

%let temps_debut = %sysfunc(time()); 
proc sql noprint;
create table match_merge_sql as 
    select a.*, b.* from large as a left join small as b on a.smallvar = b.smallvar;
quit;
%let temps_fin = %sysfunc(time());
%let duree4 = %sysevalf(&temps_fin.-&temps_debut.); 
%put Dur�e d�ex�cution : &duree4.;
proc sql noprint;
    select count(*) into :nbligne_sql from match_merge_sql;
quit;
%put nbligne_sql = &nbligne_sql.;




/* 3- jointure par hash */
/* clear log */
dm 'log;clear;';
data small; set small_sauv; run;
data large; set large_sauv; run;

%let temps_debut = %sysfunc(time()); 
data work.hash_merge (drop=rc i);
 	/* Create it */
 	declare hash h_smallq ();
 	/* Define it */
 	length  smallvar smallvar1 8 smallvar2 8;
 	/*array smallvar(20);*/
	rc = h_smallq.DefineKey ( "smallvar" );
	rc = h_smallq.DefineData ( "smallvar1","smallvar2");	
 	rc = h_smallq.DefineDone ();
 	/* Fill it */	
 	do until ( eof_small );
 		set work.small end = eof_small;
 		rc = h_smallq.add ();
 	end;
 	/* Merge it */
 	do until ( eof_large );
 		set work.large end = eof_large;	
		/* initialisation des formats des variables*/
		smallvar1=.; smallvar2=.;	
 		rc = h_smallq.find ();
		/* si la variable n'est pas trouv� alors rc != 0 */
		/*if rc ne 0 then do;
	 		smallvar1=.; smallvar2=.;
	 	end;*/ 
 		output;
 	end;
run; 
%let temps_fin = %sysfunc(time());
%let duree3 = %sysevalf(&temps_fin.-&temps_debut.); 
%put Dur�e d�ex�cution : &duree3.;
proc sql noprint;
    select count(*) into :nbligne_hash_merge from hash_merge;
quit;
%put nbligne_hash_merge = &nbligne_hash_merge.;



/* 2- jointure par index */
/* clear log */
dm 'log;clear;';
data small; set small_sauv; run;
data large; set large_sauv; run;

/*******************/
/* creating indexes uniquement sur la deuxi�me table*/
%let temps_debut = %sysfunc(time()); 
proc datasets lib=work nolist;
 modify small; index create /*keyvar*/ smallvar;
quit; 
proc datasets lib=work nolist;
 modify large; index create /*keyvar*/ smallvar;
quit; 
/* merge with index on small (large is already sorted) */
data work.match_merge_index_2;
 merge work.large (in=a)
 work.small (in=b);
 by /*keyvar*/ smallvar;
 if a;
run;
%let temps_fin = %sysfunc(time());
%let duree2_2 = %sysevalf(&temps_fin.-&temps_debut.); 
%put Dur�e d�ex�cution : &duree2_2.;
proc sql noprint;
    select count(*) into :nbligne_index2 from match_merge_index_2;
quit;
%put nbligne_index2 = &nbligne_index2.;

/********/
/********/

/* clear log */
dm 'log;clear;';
data small; set small_sauv; run;
data large; set large_sauv; run;

%let temps_debut = %sysfunc(time()); 
options msglevel=i;
/* creating indexes : creation des index pour la variable keyvar */
proc datasets lib=work nolist;
 modify small; index create /*keyvar*/ smallvar;
 modify large; index create /*keyvar*/ smallvar;
quit; 

/* jointure par index sans besoin de trier car par index*/
/* merge with indexes (no sorting) */
data work.match_merge_index;
 merge work.large (in=a)
 work.small (in=b);
 by /*keyvar*/ smallvar ;
 if a;
run; 
%let temps_fin = %sysfunc(time());
%let duree2 = %sysevalf(&temps_fin.-&temps_debut.); 
%put Dur�e d�ex�cution : &duree2.;
proc sql noprint;
    select count(*) into :nbligne_index from match_merge_index;
quit;
%put nbligne_index = &nbligne_index.;





/* 1- jointure basique */
option compress=yes;
/* clear log */
dm 'log;clear;';
data small; set small_sauv; run;
data large; set large_sauv; run;

/* basic match-merge with sort */
%let temps_debut = %sysfunc(time()); 
proc sort data=work.small; by /*keyvar*/ smallvar ; run;
proc sort data=work.large; by /*keyvar*/ smallvar ; run; 
data work.match_merge;
 	merge work.large (in=a)
 	work.small (in=b);
 	by /*keyvar*/ smallvar;
 	if a;
run;
%let temps_fin = %sysfunc(time());
%let duree1 = %sysevalf(&temps_fin.-&temps_debut.); 
%put Dur�e d�ex�cution : &duree1.;
proc sql noprint;
    select count(*) into :nbligne from match_merge;
quit;
%put nbligne = &nbligne.;


/* 1_2- jointure basique */
option compress=no;
/* clear log */
dm 'log;clear;';
data small; set small_sauv; run;
data large; set large_sauv; run;

/* basic match-merge with sort */
%let temps_debut = %sysfunc(time()); 
proc sort data=work.small; by /*keyvar*/ smallvar ; run;
proc sort data=work.large; by /*keyvar*/ smallvar ; run; 
data work.match_merge_2;
 	merge work.large (in=a)
 	work.small (in=b);
 	by /*keyvar*/ smallvar;
 	if a;
run;
%let temps_fin = %sysfunc(time());
%let duree1_2 = %sysevalf(&temps_fin.-&temps_debut.); 
%put Dur�e d�ex�cution : &duree1_2.;
proc sql noprint;
    select count(*) into :nbligne_2 from match_merge_2;
quit;
%put nbligne_2 = &nbligne_2.;



/*
data small; set small_sauv; run;
data large; set large_sauv; run;
*/

%put nbligne_sql = &nbligne_sql.;
%put nbligne_hash_merge = &nbligne_hash_merge.;
%put nbligne_index2 = &nbligne_index2.;
%put nbligne_index = &nbligne_index.;
%put nbligne = &nbligne.;
%put nbligne_2 = &nbligne_2.;

%put Dur�e d�ex�cution nbligne_sql : &duree4.;
%put Dur�e d�ex�cution nbligne_hash_merge : &duree3.;
%put Dur�e d�ex�cution nbligne_index2 : &duree2_2.;
%put Dur�e d�ex�cution nbligne_index : &duree2.;
%put Dur�e d�ex�cution nbligne : &duree1.;
%put Dur�e d�ex�cution nbligne_2 : &duree1_2.;

%mend testest; 

%testest(nb_obs_set=5000);
%testest(nb_obs_set=50000);
%testest(nb_obs_set=500000);
%testest(nb_obs_set=1000000);








/****************//****************//****************//****************//****************//****************/
/****************//****************//****************//****************//****************//****************/
/****************//****************//****************//****************//****************//****************/
data work.hash_mergeq3 (drop=rc i);
 	/* Create it */
 	declare hash h_smallq ();
 	/* Define it */
 	length  smallvar smallvar1 8 smallvar2 8;
 	/*array smallvar(20);*/
	rc = h_smallq.DefineKey ( "smallvar" );
	rc = h_smallq.DefineData ( "smallvar1","smallvar2");	
 	rc = h_smallq.DefineDone ();
 	/* Fill it */
	
 	do until ( eof_small );
 		set work.small end = eof_small;
 		rc = h_smallq.add ();
 	end;
 	/* Merge it */
 	do until ( eof_large );
 		set work.large end = eof_large;	
		/*smallvar1=.; smallvar2=.;*/	
 		rc = h_smallq.find ();
		/* si la variable n'est pas trouv� alors rc != 0 */
		if rc ne 0 then do;
	 		smallvar1=.; smallvar2=.;
	 	end; 
 		output;
 	end;
run; 



/**************************/

%let temps_debut = %sysfunc(time()); 
/* merge with memory table (no sorting or indexing required!) */
data work.hash_merge (drop=rc i);
 	/* Create it */
 	declare hash h_smallq ();
 	/* Define it */
 	length /*keyvar*/ smallvar smallvar1 8 smallvar20 8;
 	/*array smallvar(20);*/
	rc = h_smallq.DefineKey ( "smallvar" );
 	/* rc = h_smallq.DefineKey ( "keyvar" ); */
	rc = h_smallq.DefineData ( "smallvar1","smallvar2")
	/*
 	rc = h_smallq.DefineData ( "smallvar1","smallvar2","smallvar3","smallvar4",
 	"smallvar5","smallvar6","smallvar7","smallvar8",
 	"smallvar9","smallvar10","smallvar11","smallvar12",
 	"smallvar13","smallvar14","smallvar15","smallvar16",
 	"smallvar17","smallvar18","smallvar19","smallvar20" );
	*/
 	rc = h_smallq.DefineDone ();
 	/* Fill it */
 	do until ( eof_small );
 		set work.small end = eof_small;
 		rc = h_smallq.add ();
 	end;
 	/* Merge it */
 	do until ( eof_large );
 		set work.large end = eof_large;
 		/* this loop initializes variables before merging from h_smallq */
 		/*
		do i=lbound(smallvar) to hbound(smallvar);
 			smallvar(i) = .;
 		end;
		*/
 		rc = h_smallq.find ();
 		output;
 	end;
run; 
%let temps_fin = %sysfunc(time());
%let duree3 = %sysevalf(&temps_fin.-&temps_debut.); 
%put Dur�e d�ex�cution : &duree3.;
proc sql noprint;
    select count(*) into :nbligne_hash_merge from hash_merge;
quit;
%put nbligne_hash_merge = &nbligne_hash_merge.;


/*rc=object.DELETE( );*/

/******/
















/********************/
/* merge with index on small (large is already sorted) */
data work.match_merge_index;
 merge work.small (in=a)
 work.large (in=b keep=keyvar largevar1-largevar20);
 by keyvar;
 if a;
run;


/* And now using a memory table:*/
data work.hash_merge (drop=rc i);
 /* Create it */
 declare hash h_large ();
 /* Define it */
 length keyvar largevar1-largevar20 8;
 array largevar(20);
 rc = h_large.DefineKey ( "keyvar" );
 rc = h_large.DefineData ( "largevar1","largevar2","largevar3","largevar4",
 "largevar5","largevar6","largevar7","largevar8",
 "largevar9","largevar10","largevar11","largevar12",
 "largevar13","largevar14","largevar15","largevar16",
 "largevar17","largevar18","largevar19","largevar20" );
 rc = h_large.DefineDone ();
 /* Fill it */
 do until ( eof_large );
 	set work.large(keep=keyvar largevar1-largevar20) end = eof_large;
 	rc = h_large.add ();
 end;
 /* Merge it */
 do until ( eof_small );
 	set work.small end = eof_small;
 	do i=lbound(largevar) to hbound(largevar);
 		largevar(i) = .;
 	end;
 	rc = h_large.find ();
 	output;
 end;
run;


/**********/
/* simulate hash object h_small */
 proc sort data=work.small (keep=keyvar smallvar1-smallvar4)
 out=work.h_small (index=(keyvar))
 nodupkey;
 by keyvar; 

 run;

 /* corresponding hash commands */
 rc = h_small.DefineKey ( �keyvar� );
 rc = h_small.DefineData ( �smallvar1�,�smallvar2�,�smallvar3�,�smallvar4�);

 rc = h_large.DefineKey ( "keyvar",�keyseq� );
 rc = h_large.DefineData ( "largevar1","largevar2","largevar3", � )
 rc = h_large.DefineDone ();
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




/******************************************************/
/* hash */
/* Create it */
 declare hash h_small (); 
/* Define it */
 length keyvar smallvar1-smallvar4 8;
 /* recuperer le code retour dans rc. (rc=0 quand la requette s'est bien pass�e )*/
 rc = h_small.DefineKey ( "keyvar" );
 rc = h_small.DefineData ( "smallvar1","smallvar2","smallvar3","smallvar4");
 rc = h_small.DefineDone (); 
 /* Fill it : charger les donn�es avec la table work.small */
 do until ( eof_small );
 	set work.small (keep=keyvar smallvar1-smallvar4) end = eof_small;
 	rc = h_small.add ();
 end; 
 /* Access it : acc�s aux variables */
 do until ( eof_big);
 	set work.big end = eof_big;
	/* initialisation des formats des variables*/
 	smallvar1=.; smallvar2=.; smallvar3=.; smallvar4=.;
	/* chercher ces variables*/
 	rc = h_small.find ();
	/* si la variable n'est pas trouv� alors rc != 0 */
	if rc ne 0 then do;
 		smallvar1=.; smallvar2=.; smallvar3=.; smallvar4=.;
 	end; 
 	output;
 end; 


