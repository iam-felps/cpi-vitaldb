import vitaldb

tiva_list = list(vitaldb.caseids_tiva)
print(len(tiva_list))
tiva_case = tiva_list[0]
vf = vitaldb.VitalFile(tiva_case)
names = vf.get_track_names()
orchestra_related = [t for t in names if 'ORCHESTRA' in t.upper() or 'PPF' in t.upper() or 'RFTN' in t.upper()]
print("Propofol/Remifentanil-related tracks:", orchestra_related)