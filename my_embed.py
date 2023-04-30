from Bio.Seq import Seq
import json
import os
import multiprocessing




# ********* SETTINGS **********

# FILE_PATH = "./dataset/NEIS2157.json" # file containing the origina dataset. A key will be added on the dict and the file will be overwrited
# FILE_PATH = "./dataset/globins.json"
FILE_PATH = "./dataset/test.json"


ANNOTATION_KEY = "alphafold"

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ******************************



def predict(id, query_sequence):
   
    jobname = id
   
    print("Predicting", jobname)	
    import os
    import os.path
    import re
    import hashlib

    def add_hash(x,y):
      return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

    # remove whitespaces
    query_sequence = "".join(query_sequence.split())
    query_sequence = re.sub(r'[^a-zA-Z]','', query_sequence).upper()

    # remove whitespaces
    jobname = "".join(jobname.split())
    jobname = re.sub(r'\W+', '', jobname)
    jobname = add_hash(jobname, query_sequence)

    with open(f"./tmpdata/{jobname}.fasta", "w") as text_file:
        text_file.write(">1\n%s" % query_sequence)

    # number of models to use
 
    msa_mode = "single_sequence" #@param ["MMseqs2 (UniRef+Environmental)", "MMseqs2 (UniRef only)","single_sequence","custom"]
    num_models = 1 
    use_msa = False
    use_env = False
    use_custom_msa = False
    use_amber = False #@param {type:"boolean"}
    use_templates = True #@param {type:"boolean"}
 
    #@markdown ### Experimental options
    homooligomer = 1 
    save_to_google_drive = False 

    if homooligomer > 1:
      if use_amber:
        print("amber disabled: amber is not currently supported for homooligomers")
        use_amber = False
      if use_templates:
        print("templates disabled: templates are not currently supported for homooligomers")
        use_templates = False

    with open(f"./tmpdata/{jobname}.log", "w") as text_file:
        text_file.write("num_models=%s\n" % num_models)
        text_file.write("use_amber=%s\n" % use_amber)
        text_file.write("use_msa=%s\n" % use_msa)
        text_file.write("msa_mode=%s\n" % msa_mode)
        text_file.write("use_templates=%s\n" % use_templates)
        text_file.write("homooligomer=%s\n" % homooligomer)

    # decide which a3m to use
    if use_msa:
      raise Exception("MSA is not supported in this version of the notebook")
      a3m_file = f"{jobname}.a3m"
    elif use_custom_msa:
      raise Exception("Custom MSA is not supported in this version of the notebook")
      a3m_file = f"{jobname}.custom.a3m"
      if not os.path.isfile(a3m_file):
        custom_msa_dict = files.upload()
        custom_msa = list(custom_msa_dict.keys())[0]
        header = 0
        import fileinput
        for line in fileinput.FileInput(custom_msa,inplace=1):
          if line.startswith(">"):
             header = header + 1 
          if line.startswith("#"):
            continue
          if line.rstrip() == False:
            continue
          if line.startswith(">") == False and header == 1:
             query_sequence = line.rstrip() 
          print(line, end='')

        os.rename(custom_msa, a3m_file)
        print(f"moving {custom_msa} to {a3m_file}")
    else:
      a3m_file = f"./tmpdata/{jobname}.single_sequence.a3m"
      with open(a3m_file, "w") as text_file:
        text_file.write(">1\n%s" % query_sequence)
  
    # setup the model
    if "model" not in dir():

      # hiding warning messages
      import warnings
      from absl import logging
      import os

      #warnings.filterwarnings('ignore')
      logging.set_verbosity("error")
      #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'	# advanced logging
      #tf.get_logger().setLevel('ERROR')

      import sys
      import numpy as np
      from alphafold.common import protein
      from alphafold.data import pipeline
      from alphafold.data import templates
      from alphafold.model import data
      from alphafold.model import config
      from alphafold.model import model
      from alphafold.data.tools import hhsearch

      # plotting libraries

    if use_amber and "relax" not in dir():
      sys.path.insert(0, '/usr/local/lib/python3.7/site-packages/')
      from alphafold.relax import relax

    def mk_mock_template(query_sequence):
      # since alphafold's model requires a template input
      # we create a blank example w/ zero input, confidence -1
      ln = len(query_sequence)
      output_templates_sequence = "-"*ln
      output_confidence_scores = np.full(ln,-1)
      templates_all_atom_positions = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
      templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
      templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                        templates.residue_constants.HHBLITS_AA_TO_ID)
      template_features = {'template_all_atom_positions': templates_all_atom_positions[None],
                           'template_all_atom_masks': templates_all_atom_masks[None],
                           'template_sequence': [f'none'.encode()],
                           'template_aatype': np.array(templates_aatype)[None],
                           'template_confidence_scores': output_confidence_scores[None],
                           'template_domain_names': [f'none'.encode()],
                           'template_release_date': [f'none'.encode()]}
      return template_features

    def mk_template(jobname):
      template_featurizer = templates.TemplateHitFeaturizer(
          mmcif_dir="templates/",
          max_template_date="2100-01-01",
          max_hits=20,
          kalign_binary_path="kalign",
          release_dates_path=None,
          obsolete_pdbs_path=None)

      hhsearch_pdb70_runner = hhsearch.HHSearch(binary_path="hhsearch",databases=[jobname])

      a3m_lines = "\n".join(open(f"{jobname}.a3m","r").readlines())
      hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
      hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
      templates_result = template_featurizer.get_templates(query_sequence=query_sequence,
                                                           query_pdb_code=None,
                                                           query_release_date=None,
                                                           hits=hhsearch_hits)
      return templates_result.features

    def set_bfactor(pdb_filename, bfac, idx_res, chains):
      I = open(pdb_filename,"r").readlines()
      O = open(pdb_filename,"w")
      for line in I:
        if line[0:6] == "ATOM  ":
          seq_id = int(line[22:26].strip()) - 1
          seq_id = np.where(idx_res == seq_id)[0][0]
          O.write(f"{line[:21]}{chains[seq_id]}{line[22:60]}{bfac[seq_id]:6.2f}{line[66:]}")
      O.close()

    def predict_structure(prefix, feature_dict, Ls, model_params, use_model, do_relax=False, random_seed=0):  
      """Predicts structure using AlphaFold for the given sequence."""

      # Minkyung's code
      # add big enough number to residue index to indicate chain breaks
      idx_res = feature_dict['residue_index']
      L_prev = 0
      # Ls: number of residues in each chain
      for L_i in Ls[:-1]:
          idx_res[L_prev+L_i:] += 200
          L_prev += L_i  
      chains = list("".join([ascii_uppercase[n]*L for n,L in enumerate(Ls)]))
      feature_dict['residue_index'] = idx_res

      # Run the models.
      plddts,paes = [],[]
      unrelaxed_pdb_lines = []
      relaxed_pdb_lines = []

      for model_name, params in model_params.items():
        if model_name in use_model:
          print(f"running {model_name}")
          # swap params to avoid recompiling
          # note: models 1,2 have diff number of params compared to models 3,4,5
          if any(str(m) in model_name for m in [1,2]): model_runner = model_runner_1
          if any(str(m) in model_name for m in [3,4,5]): model_runner = model_runner_3
          model_runner.params = params

          processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
          prediction_result = model_runner.predict(processed_feature_dict)
          unrelaxed_protein = protein.from_prediction(processed_feature_dict,prediction_result)
          unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
          plddts.append(prediction_result['plddt'])
          paes.append(prediction_result['predicted_aligned_error'])

          if do_relax:
            # Relax the prediction.
            amber_relaxer = relax.AmberRelaxation(max_iterations=0,tolerance=2.39,
                                                  stiffness=10.0,exclude_residues=[],
                                                  max_outer_iterations=20)      
            relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
            relaxed_pdb_lines.append(relaxed_pdb_str)

      # rerank models based on predicted lddt
      lddt_rank = np.mean(plddts,-1).argsort()[::-1]
      out = {}
      print("reranking models based on avg. predicted lDDT")
      for n,r in enumerate(lddt_rank):
        print(f"model_{n+1} {np.mean(plddts[r])}")

        unrelaxed_pdb_path = f'./tmpdata/{prefix}_unrelaxed_model_{n+1}.pdb'    
        with open(unrelaxed_pdb_path, 'w') as f: f.write(unrelaxed_pdb_lines[r])
        set_bfactor(unrelaxed_pdb_path, plddts[r]/100, idx_res, chains)

        if do_relax:
          relaxed_pdb_path = f'{prefix}_relaxed_model_{n+1}.pdb'
          with open(relaxed_pdb_path, 'w') as f: f.write(relaxed_pdb_lines[r])
          set_bfactor(relaxed_pdb_path, plddts[r]/100, idx_res, chains)

        out[f"model_{n+1}"] = {"plddt":plddts[r], "pae":paes[r]}
      return out, prediction_result
    #@title Gather input features, predict structure { vertical-output: true }
    from string import ascii_uppercase
    #os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # collect model weights
    use_model = {}
    if "model_params" not in dir(): model_params = {}
    for model_name in ["model_1","model_2","model_3","model_4","model_5"][:num_models]:
      use_model[model_name] = True
      if model_name not in model_params:
        model_params[model_name] = data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir=".")
        if model_name == "model_1":
          model_config = config.model_config(model_name+"_ptm")
          model_config.data.eval.num_ensemble = 1
          model_runner_1 = model.RunModel(model_config, model_params[model_name])
        if model_name == "model_3":
          model_config = config.model_config(model_name+"_ptm")
          model_config.data.eval.num_ensemble = 1
          model_runner_3 = model.RunModel(model_config, model_params[model_name])

    # parse TEMPLATES
    if use_templates and os.path.isfile(f"{jobname}_hhm.ffindex"):
      template_features = mk_template(jobname)
    else:
      template_features = mk_mock_template(query_sequence * homooligomer)

    # parse MSA
    a3m_lines = "".join(open(a3m_file,"r").readlines())
    msa, deletion_matrix = pipeline.parsers.parse_a3m(a3m_lines)

    if homooligomer == 1:
      msas = [msa]
      deletion_matrices = [deletion_matrix]
    else:
      # make multiple copies of msa for each copy
      # AAA------
      # ---AAA---
      # ------AAA
      #
      # note: if you concat the sequences (as below), it does NOT work
      # AAAAAAAAA
      msas = []
      deletion_matrices = []
      Ln = len(query_sequence)
      for o in range(homooligomer):
        L = Ln * o
        R = Ln * (homooligomer-(o+1))
        msas.append(["-"*L+seq+"-"*R for seq in msa])
        deletion_matrices.append([[0]*L+mtx+[0]*R for mtx in deletion_matrix])

    # gather features
    feature_dict = {
        **pipeline.make_sequence_features(sequence=query_sequence*homooligomer,
                                          description="none",
                                          num_res=len(query_sequence)*homooligomer),
        **pipeline.make_msa_features(msas=msas,deletion_matrices=deletion_matrices),
        **template_features
    }
    outs, prediction_result = predict_structure(jobname, feature_dict,
                             Ls=[len(query_sequence)]*homooligomer,
                             model_params=model_params, use_model=use_model,
                             do_relax=use_amber)
    
    embed = prediction_result['representations']['single']

    with open(FILE_PATH, "r") as file:   # load the list of seqrecords alreay annotated with the others embeddings
        seq_dict = json.load(file)
        seq_dict[id][ANNOTATION_KEY] = embed.tolist()
    
    with open(FILE_PATH, "w") as file:   # save the list of seqrecords alreay annotated with the others embeddings
        json.dump(seq_dict, file, indent=4)

    print("element added")
    return




def main():
    # remove all the file in the tmpdata folder
	for file in os.listdir("./tmpdata"):
		os.remove(os.path.join("./tmpdata", file))

    # if the folder params is not present, download the params
	if not os.path.isdir("params"):
		raise Exception("The folder params is not present. Please download the params and put them in the folder params")
		""" 
		!sed -i "s/pdb_lines.append('END')//" ./alphafold/common/protein.py
		!sed -i "s/pdb_lines.append('ENDMDL')//" ./alphafold/common/protein.py
		!wget -qnc https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar
		!mkdir params
		!tar -xf alphafold_params_2021-07-14.tar -C params/
		!rm alphafold_params_2021-07-14.tar
		"""

	seq_dict = []
    
	with open(FILE_PATH, "r") as file:   # load the list of seqrecords alreay annotated with the others embeddings
		seq_dict = json.load(file)

	
	for id in seq_dict.keys():
		seq_string = seq_dict[id]["sequence"]

		seq_string = seq_string.replace(" ", "").replace("\n", "")

		if set(seq_string).issubset(set(["A", "C", "G", "T"])):
			seq_string = str(Seq(seq_string).translate(stop_symbol=""))
			print("The nucleotides sequence for ", id, " has been translated")

		print("Predicting the embedding for ", id, "...")

		# the code run in a different process to avoid memory leaks

		p = multiprocessing.Process(target=predict, args=(id, seq_string, ))
		p.start()

		p.join()

		print('done')

		
        


if __name__ == "__main__":
    main()
