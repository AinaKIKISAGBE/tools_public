
/* add new column "ipPublic" in iptable */
ALTER TABLE IF EXISTS public.iptable
    ADD COLUMN "ipPublic" text;

	
	
/* selecte where column is null*/
SELECT * FROM public.iptable WHERE jobrunformat is null


/* add new row in tb1keys */
INSERT INTO public.tb1keys (id, keys_source, keys_value, keys_url) 
VALUES(4,'SURFSHARK_key1',
	   '{"SURFSHARK_USER":"hvRCqukpWz3JXvR7PJPLShF4",
"SURFSHARK_PASSWORD":"CFkPatwF27Xz9tBXhwcDPJvA",
"SURFSHARK_COUNTRY":"fr",
"SURFSHARK_CITY":"" ,
"OPENVPN_OPTS":"" ,
"CONNECTION_TYPE":"tcp",
"LAN_NETWORK":"" ,
"CREATE_TUN_DEVICE":"" ,
"OVPN_CONFIGS":"" ,
"ENABLE_KILL_SWITCH":"true"}',
	   'https://surfshark.com/fr');
	   
	   
	
	