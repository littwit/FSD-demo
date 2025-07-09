#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
# import numpy as np
import zipfile
import pandas as pd
from io import TextIOWrapper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

pth = os.path.dirname(os.path.abspath(__file__))

def load_csvs_from_zip(zip_path):
    """
    Read and concatenate CSV files from subfolders in a ZIP file.
    Files within each subfolder are combined into a single DataFrame.
    
    Parameters:
        zip_path (str): Path to the ZIP file
        
    Returns:
        dict: Structure as {subfolder_name: concatenated_DataFrame}
    """
    result = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # First pass: collect all CSV files by subfolder
        csv_files = {}
        
        for file_info in zip_ref.infolist():
            if file_info.is_dir() or not file_info.filename.lower().endswith('.csv'):
                continue
                
            dirname, filename = os.path.split(file_info.filename)
            subfolder = os.path.basename(dirname) if dirname else 'root'
            
            if subfolder not in csv_files:
                csv_files[subfolder] = []
            csv_files[subfolder].append(file_info.filename)
        
        # Second pass: read and concatenate CSVs per subfolder
        for subfolder, files in csv_files.items():
            dfs = []
            
            for file_path in files:
                try:
                    with zip_ref.open(file_path) as csv_file:
                        text_file = TextIOWrapper(csv_file, encoding='utf-8')
                        df = pd.read_csv(text_file)
                        dfs.append(df)
                except Exception as e:
                    print(f"Failed to read {file_path}: {str(e)}")
            
            if dfs:  # Only concatenate if we successfully read at least one file
                result[subfolder] = pd.concat(dfs, ignore_index=True)
            else:
                result[subfolder] = None
                print(f"No valid CSV files found in {subfolder}")
                    
    return result

def VisualFullFields(data):
    
    C = 9
    N = 5
    id = [10, 15, 20, 25, 30] # visual time steps
    res = data[id, :, 2:] # save coordinates & physical fields

    ax_min, ax_max = np.amin(res[:, :, 2:], axis=(0, 1)) , np.amax(res[:, :, 2:], axis=(0, 1)) 
    
    items = ['(℃)', '(%)',  '(%)','(%)','(%)','(MPa)', '(MPa)','(MPa)','(MPa)']
    row_titles = [r'$T$', r'$\epsilon_{r}$', r'$\epsilon_{z}$', r'$\epsilon_{\theta}$',r'$\gamma_{rz}$',
                    r'$\sigma_{r}$', r'$\sigma_{z}$', r'$\sigma_{\theta}$',r'$\tau_{rz}$' ]
    
    plt.clf()
    plt.rcParams.update({'font.size':9}) 
    fig, axes = plt.subplots(C, N, figsize=(7, 8.5),  
                             constrained_layout=True
                             )
    scale = 1.0
    
    for i in range(C):
        vmin, vmax = ax_min[i], ax_max[i]
        cmap='viridis'
        for j in range(N):
            im = axes[i, j].tripcolor(res[j, :, 0], res[j, :, 1], res[j, :, i+2], 
                                    #    s=3,
                                        cmap=cmap, 
                                        vmin=vmin, 
                                        vmax=vmax, 
                                        shading='gouraud',
                                        # levels=20
                                        edgecolors='none'
                                        
                                        )

            # if j == N-1:
            if i == C-1:
                axes[i, j].set_xlabel(r'$r$'+ ' (mm)')
                
            axes[i, j].set_xticks(np.linspace(0, 30, 4),)
            # axes[i, 0].set_xticklabels(x1)
            # ax[i, 0].set_xlabel('x')
            axes[i, j].set_yticks(np.linspace(0, 30, 4),)
            # axes[i, 0].set_yticklabels(y1)
            if j == 0:
                axes[i, j].set_ylabel(f'{row_titles[i]}\n'+r'$z$' + ' (mm)')
                
            axes[i, j].set_aspect(scale)
            
            if i == 0:
                axes[i, j].set_title(f's={0.5*(id[j]) } mm\n') 
            
            axes[i, j].set_ylim(0, 30)
            axes[i, j].set_xlim(0, 30)

        cbar = fig.colorbar(im, 
                            ax=[axes[i, m] for m in range(N)],
                            fraction=0.046,
                            pad=0.04, 
                            shrink=1.0, 
                            )
        # cbar1.set_label(items[i])

        cbar.ax.tick_params(labelsize=6,)
        
        tick1 = np.linspace(vmin, vmax, 5)
        # tick1 = [ for tick in tick1]
        cbar.set_ticks(tick1)
        cbar.set_label(items[i], rotation=90, labelpad=5, fontsize=8)
        
        # tick1 = tick1.tolist()
        # tick1[2] = items[i]
        cbar.set_ticklabels(tick1)
        for k in cbar.ax.get_yticklabels():
            if k.get_text() in items:
                # k.set_rotation(90)
                k.set_va('center')
        cbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))      
    plt.savefig(f'{pth}/Visual_full_field.pdf', dpi=900)



# Usage example
if __name__=='__main__':

    data = load_csvs_from_zip(f'{pth}/SimuRes.zip')

    # Access CSV data in specific subfolders
    for subfolder, csv_dict in data.items():
        if csv_dict is not None:
            print(f"\nSubfolder: {subfolder}")
            d = csv_dict.values[:, 1:-1].reshape(-1, 13)
            # sort by time_step
            sort_id = np.argsort(d[:, 0])
            d = d[sort_id] 
            d = d.reshape(-1, 1080, 13) # (time_step, 1080, 13)
            
            # visual full field
            VisualFullFields(d)
            pass
