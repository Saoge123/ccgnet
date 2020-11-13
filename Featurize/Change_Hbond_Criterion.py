# -*- coding: utf-8 -*-
import ccdc

"""
Default H-Bond criterion:
  ccdc.molecule.Molecule.HBondCriterion().acceptor_types.items():
    [(u'+nitrogen', 'NEVER'),
     (u'metal bound N', 'YES'),
     (u'terminal N (cyano, etc.)', 'YES'),
     (u'aromatic (6-ring) N', 'YES'),
     (u'other 2-coordinate N', 'YES'),
     (u'3-coordinate N', 'YES'),
     (u'unclassified N', 'YES'),
     (u'-nitrogen', 'NEVER'),
     (u'+oxygen', 'NEVER'),
     (u'metal bound O', 'YES'),
     (u'carboxylate O', 'YES'),
     (u'other terminal O (C=O, S=O ...)', 'YES'),
     (u'water O', 'YES'),
     (u'hydroxyl O', 'YES'),
     (u'bridging O (ether, etc.)', 'YES'),
     (u'unclassified O', 'YES'),
     (u'-oxygen', 'NEVER'),
     (u'+sulphur', 'NEVER'),
     (u'metal bound S', 'YES'),
     (u'terminal S', 'YES'),
     (u'unclassified S', 'YES'),
     (u'-sulphur', 'NEVER'),
     (u'+fluorine', 'NEVER'),
     (u'metal bound F', 'YES'),
     (u'fluoride ion (F-)', 'YES'),
     (u'unclassified F', 'NO'),
     (u'-fluorine', 'NEVER'),
     (u'+chlorine', 'NEVER'),
     (u'metal bound Cl', 'YES'),
     (u'chloride ion (Cl-)', 'YES'),
     (u'unclassified Cl', 'YES'),
     (u'-chlorine', 'NEVER'),
     (u'+bromine', 'NEVER'),
     (u'metal bound Br', 'YES'),
     (u'bromide ion (Br-)', 'YES'),
     (u'unclassified Br', 'YES'),
     (u'-bromine', 'NEVER'),
     (u'+iodine', 'NEVER'),
     (u'metal bound I', 'YES'),
     (u'iodide ion (I-)', 'YES'),
     (u'unclassified I', 'YES'),
     (u'-iodine', 'NEVER')]
  ccdc.molecule.Molecule.HBondCriterion().donnor_types.items():
    [(u'+nitrogen', 'NEVER'),
     (u'metal bound N', 'YES'),
     (u'imine N', 'YES'),
     (u'aromatic (6-ring) N', 'YES'),
     (u'amide or thioamide N', 'YES'),
     (u'planar N', 'YES'),
     (u'pyramidal N', 'YES'),
     (u'ammonium N (NH4+, RNH3+ etc.)', 'YES'),
     (u'unclassified  N', 'YES'),
     (u'-nitrogen', 'NEVER'),
     (u'+oxygen', 'NEVER'),
     (u'metal bound O', 'YES'),
     (u'water O', 'YES'),
     (u'hydroxyl O', 'YES'),
     (u'unclassified  O', 'YES'),
     (u'-oxygen', 'NEVER'),
     (u'+sulphur', 'NEVER'),
     (u'metal bound S', 'YES'),
     (u'non metal bound S', 'YES'),
     (u'-sulphur', 'NEVER'),
     (u'+carbon', 'NEVER'),
     (u'sp C', 'NO'),
     (u'sp2 C', 'NO'),
     (u'sp3 C', 'NO'),
     (u'aromatic C', 'NO'),
     (u'carboncationic C', 'NO'),
     (u'-carbon', 'NEVER')]
"""
donnor_change_list = [(u'sp C', True),(u'sp2 C', True),(u'sp3 C', True),(u'aromatic C', True),(u'carboncationic C', True)]
acceptor_change_list = []
#criterion = ccdc.molecule.Molecule.HBondCriterion()
def Change_Hbond_Criterion( 
             criterion=ccdc.molecule.Molecule.HBondCriterion(),
             donnor_types_to_change=donnor_change_list, 
             acceptor_types_to_change=acceptor_change_list):
    if donnor_types_to_change:
        for i in donnor_types_to_change:
            criterion.donor_types[i[0]] = i[1]
    if acceptor_types_to_change:
        for j in acceptor_types_to_change:
            criterion.acceptor_types[j[0]] = j[1]
    return criterion