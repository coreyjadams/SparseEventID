import ROOT
import numpy
from ROOT import larcv
import pandas

import argparse

def init_file(file_name):
    io_manager = larcv.IOManager()
    io_manager.add_in_file(file_name)
    io_manager.initialize()
    return io_manager



def convert_to_pandas(io_manager, mode='split', n_entries=None):
    
    # This function reads in the data from larcv, and converts to pandas.  It stores the neutrino energy (true),
    # as well as the true and predicted labels.
    
    # initialize a dataframe:
    df = pandas.DataFrame(
        index = numpy.arange(io_manager.get_n_entries()),
        columns = ["pred_neut", "pred_neut0", "pred_neut1", "pred_neut2",
                   "pred_npi",  "pred_npi0",  "pred_npi1",
                   "pred_cpi",  "pred_cpi0",  "pred_cpi1",
                   "pred_prot", "pred_prot0", "pred_prot1", "pred_prot2",
                   "true_neut", "true_npi", "true_cpi", "true_prot",
                   "true_mult", "pred_mult",
                   "energy", "pot"]
    )
    
    n_written = -1

    for i in range(io_manager.get_n_entries()):

        if n_entries is not None:
            if i >= n_entries:
                break;

        io_manager.read_entry(i)
#         print ("Event ", i)
        
        # get the true labels:
        cpiID  = io_manager.get_data("particle", "cpiID")
        npiID  = io_manager.get_data("particle", "npiID")
        neutID = io_manager.get_data("particle", "neutID")
        protID = io_manager.get_data("particle", "protID")
        allID  = io_manager.get_data("particle", "all")
        neutrino = io_manager.get_data("particle", "sbndneutrino")
        
    #     if i != 0:
    #         df.iloc[i-1]['true_cpi']  = cpiID.as_vector().front().pdg_code()
    #         df.iloc[i-1]['true_npi']  = npiID.as_vector().front().pdg_code()
    #         df.iloc[i-1]['true_neut'] = neutID.as_vector().front().pdg_code()
    #         df.iloc[i-1]['true_prot'] = protID.as_vector().front().pdg_code()
    #         df.iloc[i-1]['true_mult'] = allID.as_vector().front().pdg_code()
    #         df.iloc[i-1]['energy']    = neutrino.as_vector().front().energy_init()

    #         # pot is per event, by truth information:
    #         if neutID.as_vector().front().pdg_code() == 0:
    #             # nueCC
    #             df.iloc[i-1]['pot'] = 1.99e16
    #         elif neutID.as_vector().front().pdg_code() == 1:
    #             # numuCC
    #             df.iloc[i-1]['pot'] = 1.83e14
    #         else:  # neutID.as_vector().front().pdg_code() == 2
    #             # NC
    #             df.iloc[i-1]['pot'] = 5.19e14

    #     if mode == "split":
    #         label_cpi  = io_manager.get_data("meta", "label_cpi")
    #         label_npi  = io_manager.get_data("meta", "label_npi")
    #         label_prot = io_manager.get_data("meta", "label_prot")
    #         label_neut = io_manager.get_data("meta", "label_neut")

    # #         df.iloc[i]['pred_cpi'] = label_cpi.as_vector().front().pdg_code()
    #         logits_cpi  = label_cpi.get_darray('meta')
    #         logits_npi  = label_npi.get_darray('meta')
    #         logits_prot = label_prot.get_darray('meta')
    #         logits_neut = label_neut.get_darray('meta')

    #         index = i

    #         if i != io_manager.get_n_entries() -1 :
    #             df.iloc[index]["pred_neut"]  = numpy.argmax(logits_neut)
    #             df.iloc[index]["pred_neut0"] = logits_neut[0]
    #             df.iloc[index]["pred_neut1"] = logits_neut[1]
    #             df.iloc[index]["pred_neut2"] = logits_neut[2]

    #             df.iloc[index]["pred_cpi"]  = numpy.argmax(logits_cpi)
    #             df.iloc[index]["pred_cpi0"] = logits_cpi[0]
    #             df.iloc[index]["pred_cpi1"] = logits_cpi[1]

    #             df.iloc[index]["pred_npi"]  = numpy.argmax(logits_npi)
    #             df.iloc[index]["pred_npi0"] = logits_npi[0]
    #             df.iloc[index]["pred_npi1"] = logits_npi[1]

    #             df.iloc[index]["pred_prot"]  = numpy.argmax(logits_prot)
    #             df.iloc[index]["pred_prot0"] = logits_prot[0]
    #             df.iloc[index]["pred_prot1"] = logits_prot[1]
    #             df.iloc[index]["pred_prot2"] = logits_prot[2]
        

        df.iloc[i]['true_cpi']  = cpiID.as_vector().front().pdg_code()
        df.iloc[i]['true_npi']  = npiID.as_vector().front().pdg_code()
        df.iloc[i]['true_neut'] = neutID.as_vector().front().pdg_code()
        df.iloc[i]['true_prot'] = protID.as_vector().front().pdg_code()
        df.iloc[i]['true_mult'] = allID.as_vector().front().pdg_code()
        df.iloc[i]['energy']    = neutrino.as_vector().front().energy_init()

        # pot is per event, by truth information:
        if neutID.as_vector().front().pdg_code() == 0:
            # nueCC
            df.iloc[i]['pot'] = 1.99e16
        elif neutID.as_vector().front().pdg_code() == 1:
            # numuCC
            df.iloc[i]['pot'] = 1.83e14
        else:  # neutID.as_vector().front().pdg_code() == 2
            # NC
            df.iloc[i]['pot'] = 5.19e14

        if mode == "split":
            label_cpi  = io_manager.get_data("meta", "label_cpi")
            label_npi  = io_manager.get_data("meta", "label_npi")
            label_prot = io_manager.get_data("meta", "label_prot")
            label_neut = io_manager.get_data("meta", "label_neut")

    #         df.iloc[i]['pred_cpi'] = label_cpi.as_vector().front().pdg_code()
            logits_cpi  = label_cpi.get_darray('meta')
            logits_npi  = label_npi.get_darray('meta')
            logits_prot = label_prot.get_darray('meta')
            logits_neut = label_neut.get_darray('meta')

            index = i
            df.iloc[index]["pred_neut"]  = numpy.argmax(logits_neut)
            df.iloc[index]["pred_neut0"] = logits_neut[0]
            df.iloc[index]["pred_neut1"] = logits_neut[1]
            df.iloc[index]["pred_neut2"] = logits_neut[2]

            df.iloc[index]["pred_cpi"]  = numpy.argmax(logits_cpi)
            df.iloc[index]["pred_cpi0"] = logits_cpi[0]
            df.iloc[index]["pred_cpi1"] = logits_cpi[1]

            df.iloc[index]["pred_npi"]  = numpy.argmax(logits_npi)
            df.iloc[index]["pred_npi0"] = logits_npi[0]
            df.iloc[index]["pred_npi1"] = logits_npi[1]

            df.iloc[index]["pred_prot"]  = numpy.argmax(logits_prot)
            df.iloc[index]["pred_prot0"] = logits_prot[0]
            df.iloc[index]["pred_prot1"] = logits_prot[1]
            df.iloc[index]["pred_prot2"] = logits_prot[2]
        


        else:
            pass

        n_written += 1


    # Normalize the weights.
    # We normalize everything to 1e20 POT.  So, sum the POT of each type of neutrino (true)
    # interaction and set the weight so that it sums to 1e20


    return df[0:n_written]


# This quickly calculates the accuracy:
def calculate_accuracy(df):
    for key in ['neut', 'npi', 'cpi', 'prot']:
        accuracy = numpy.mean(df['true_{}'.format(key)] == df['pred_{}'.format(key)])
        print("{0}: {1:.3}%".format(key, 100*accuracy))


def main():

    parser = argparse.ArgumentParser(description="Configuration Flags")
    parser.add_argument('-f', '--file', type=str, help="Name of larcv file", required=True)

    args = parser.parse_args()

    print(args)
    
    io_manager = init_file(args.file)
    df = convert_to_pandas(io_manager, n_entries=None)


    calculate_accuracy(df)

    out_file = args.file.replace(".root", ".pd")
    df.to_pickle(out_file)


if __name__ == "__main__":
    main()




