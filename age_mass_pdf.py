import read_mist_models as rmm
from matplotlib import pyplot as plt
import csv
import math
import glob
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
import pandas as pd
from matplotlib.collections import LineCollection
import pathlib
import scipy.stats as stats

fontsize=15

#Example isochrone file name: 300Myr_1800Myr_50Myr_step.chr
hist_only=False

def main():
    profiles = glob.glob('Star_Profiles/*.sp')
    for profile in profiles:
        print(profile)
        Read the star profile file
        star = read_star_profile(profile)
    print(star.evo)
    res_dir = 'Results/{}'.format(star.name)
    res_fn = 'Results/{}/{}.res'.format(star.name,star.name)
    
    pathlib.Path(res_dir).mkdir(parents=True, exist_ok=True)
    
    sig_limit=10000
    
    print('=========================================================================')
    run_age_calc(star,res_fn,star.mass_range,star.age_range,star.age_res,sig_limit,star.hist_age_res)
    run_mass_calc(star,res_fn,star.mass_range,star.mass_res,star.age_range,sig_limit,star.hist_mass_res)
        
def run_age_calc(star,res_fn,mass_range,age_range,age_res,sig_limit,hist_age_res):
    mt_files = sorted(glob.glob('{}/*eep'.format(star.evo)))  #All files for the given metallicity
    
    hist_age_bins = define_hist_age_bins(age_range,hist_age_res)
    
    first_mt = True
    
    plot_xs = []
    plot_ys = []
    plot_ms = []
    
    #Build the dataframe with logt, logl, age, mass
    for mtfn in mt_files:
        mt = rmm.EEP(mtfn)
        if mt.minit >= mass_range[0] and mt.minit <= mass_range[1]:
            print('In range for Mass: {} M_sun'.format(mt.minit))
            #plt.plot(mt.eeps['log_Teff'],mt.eeps['log_L'],'k-',label=mt.minit)
            px = mt.eeps['log_Teff']
            py = mt.eeps['log_L']
            px = limit_between(px,mt.eeps['star_age'],np.array(age_range)*1e6)
            py = limit_between(py,mt.eeps['star_age'],np.array(age_range)*1e6)
            plot_xs.append(px)
            plot_ys.append(py)
            plot_ms.append(mt.minit)
            mt_logt,mt_logl,ages = interpolate_mt(mt,age_range,age_res)
            data = {'logt':mt_logt,'logl':mt_logl,'age':ages}
            if first_mt:
                track_df = pd.DataFrame(data=data)
                track_df['mass']=mt.minit
                first_mt=False
            else:
                this_df = pd.DataFrame(data=data)
                this_df['mass']=mt.minit
                track_df = pd.concat([track_df,this_df],ignore_index=True,axis=0)
    
    #Plot the mass tracks
    lc = multiline(plot_xs,plot_ys,plot_ms,cmap='gist_rainbow')
    cbar = plt.colorbar(lc)
    cbar.set_label('Initial Model Mass (M_sun)')
        
    track_df['sigt']=get_sig(track_df.logt,star.logt,star.logterrlo,star.logterrhi)
    track_df['sigl']=get_sig(track_df.logl,star.logl,star.loglerrlo,star.loglerrhi)
    track_df['sig'] = np.sqrt(np.array(track_df.sigt)**2+np.array(track_df.sigl)**2)
    track_df['gauss'] = [get_gauss(sig) for sig in track_df.sig]
    
    
    in_sig_df=track_df[track_df.sig < sig_limit]
    
    print(in_sig_df)
    
    #plt.scatter(in_sig_df.logt,in_sig_df.logl,s=1,c=in_sig_df.gauss,cmap='Blues',zorder=3,alpha=0.2)
    
    plot_terr = ([star.logterrlo],[star.logterrhi])
    plot_lerr = ([star.loglerrlo],[star.loglerrhi])
    plot_rerr = ([star.logrerrlo],[star.logrerrhi])
    
    plt.errorbar(star.logt,star.logl,xerr=plot_terr,yerr=plot_lerr,elinewidth=4,color='black')
    
    
    plt.xlabel('log(Teff) [K]', fontsize=fontsize)
    plt.ylabel('log(L/Lsun)', fontsize=fontsize)
    
    
    #For full hrd
    plt.xlim(5.5,3)
    plt.ylim(-1,5)
    
    if hist_only == False:
        plt.savefig('Results/{}/{}_MT_HRD_full.png'.format(star.name,star.name))

    #For zoom out
    plt.xlim(star.logt+0.1,star.logt-0.1)
    plt.ylim(star.logl-0.3,star.logl+0.3)
    
    #plt.show()
    if hist_only == False:
        plt.savefig('Results/{}/{}_MT_HRD_zoomout.png'.format(star.name,star.name))
    
    #For zoom out
    plt.xlim(star.logt+0.02,star.logt-0.02)
    plt.ylim(star.logl-0.08,star.logl+0.08)
    if hist_only == False:
        plt.savefig('Results/{}/{}_MT_HRD.png'.format(star.name,star.name))
    plt.close()
    
    age_details_file = res_fn+'.age'
    
    #Do all masses
    open(res_fn,'w').write('Results for age for all masses:\n')
    print('Results for age for all masses:')
    age_report,age_med,age_sig1,age_sig2 = make_report(in_sig_df,'age')
    print(age_report)
    open(res_fn,'a').write(age_report)
    
    plt.hist(in_sig_df.age/1e6,weights=in_sig_df.gauss,density=True,bins=hist_age_bins)
    plt.xlabel('Age (Myr)')
    plt.ylabel('Probability Density')
    plt.xlim(hist_age_bins[0],hist_age_bins[-1])
    #plt.axvline(age_med, color='k')
    #plt.axvline(age_sig1[0], color='k',linestyle='dashed')
    plt.axvline(age_sig1[1], color='k',linestyle='dashed')
    #plt.axvline(age_sig2[0], color='k',linestyle='dotted')
    plt.axvline(age_sig2[1], color='k',linestyle='dotted')
    plt.savefig('Results/{}/{}_age_pdf.png'.format(star.name,star.name))
    plt.close()
    
def run_mass_calc(star,res_fn,mass_range,mass_res,age_range,sig_limit,hist_mass_res):
    iso_files = sorted(glob.glob('{}/*chr'.format(star.evo)))  #All files for the given metallicity
    
    hist_mass_bins = define_hist_mass_bins(mass_range,hist_mass_res)

    print('Iso file(s):\n',iso_files)
    
    iso_df = build_iso_df(iso_files,mass_range,mass_res,age_range)
    
    iso_df['sigt']=get_sig(iso_df.logt,star.logt,star.logterrlo,star.logterrhi)
    iso_df['sigl']=get_sig(iso_df.logl,star.logl,star.loglerrlo,star.loglerrhi)
    iso_df['sig'] = np.sqrt(np.array(iso_df.sigt)**2+np.array(iso_df.sigl)**2)
    iso_df['gauss'] = [get_gauss(sig) for sig in iso_df.sig]
    
    in_sig_df=iso_df[iso_df.sig < sig_limit]
    print(in_sig_df)
    
    #plt.scatter(in_sig_df.logt,in_sig_df.logl,s=1,c=in_sig_df.gauss,cmap='Blues',zorder=3,alpha=0.2)
    
    plot_terr = ([star.logterrlo],[star.logterrhi])
    plot_lerr = ([star.loglerrlo],[star.loglerrhi])
    plot_rerr = ([star.logrerrlo],[star.logrerrhi])
    
    plt.errorbar(star.logt,star.logl,xerr=plot_terr,yerr=plot_lerr,elinewidth=4,color='black')
    
    plt.xlabel('log(Teff) [K]', fontsize=fontsize)
    plt.ylabel('log(L/Lsun)', fontsize=fontsize)
    
    #For full hrd
    plt.xlim(5.5,3)
    plt.ylim(-1,5)
    if hist_only == False:
        plt.savefig('Results/{}/{}_ISO_HRD_full.png'.format(star.name,star.name))
    
    #For zoom out
    plt.xlim(star.logt+0.1,star.logt-0.1)
    plt.ylim(star.logl-0.3,star.logl+0.3)
    if hist_only == False:
        plt.savefig('Results/{}/{}_ISO_HRD_zoomout.png'.format(star.name,star.name))
    
    #For zoom out
    plt.xlim(star.logt+0.02,star.logt-0.02)
    plt.ylim(star.logl-0.08,star.logl+0.08)
    if hist_only == False:
        plt.savefig('Results/{}/{}_ISO_HRD.png'.format(star.name,star.name))
    plt.close()
    
    
    mass_details_file = res_fn+'.mass'
    
    '''
    #Do individual ages
    open(mass_details_file,'w').write('Results for mass by age:\n')
    #print('Results for mass by age:')
    for a in in_sig_df.age.unique():
        this_df = in_sig_df[in_sig_df.age == a]
        #print('{} Myr'.format(int(a/1e6)))
        open(mass_details_file,'a').write('{} Myr\n'.format(int(a/1e6)))
        plt.hist(this_df.star_mass,weights=this_df.gauss,density=True,bins=hist_mass_bins)
        plt.xlabel('Mass (M_sun)')
        plt.ylabel('Normalized probability')
        plt.xlim(hist_mass_bins[0],hist_mass_bins[-1])
        plt.savefig('Results/{}/{}_mass_pdf_A{}.png'.format(star.name,star.name,int(a/1e6)))
        plt.close()
        mass_report = make_report(this_df,'mass')
        #print(mass_report)
        open(mass_details_file,'a').write(mass_report)
        open(mass_details_file,'a').write('----------------------------\n')
    '''
    
    #Do all ages
    open(res_fn,'a').write('----------------------------\n')
    open(res_fn,'a').write('Results for mass for all ages:\n')
    print('Results for mass for all ages:')
    mass_report,mass_med,mass_sig1,mass_sig2 = make_report(in_sig_df,'mass')
    print(mass_report)
    open(res_fn,'a').write(mass_report)
    plt.hist(in_sig_df.star_mass,weights=in_sig_df.gauss,density=True,label='Mass (M_sun)',bins=hist_mass_bins)
    plt.xlabel('Mass (M_sun)')
    plt.ylabel('Probability Density')
    plt.xlim(hist_mass_bins[0],hist_mass_bins[-1])
    plt.axvline(mass_med, color='k')
    plt.axvline(mass_sig1[0], color='k',linestyle='dashed')
    plt.axvline(mass_sig1[1], color='k',linestyle='dashed')
    plt.axvline(mass_sig2[0], color='k',linestyle='dotted')
    plt.axvline(mass_sig2[1], color='k',linestyle='dotted')
    plt.savefig('Results/{}/{}_mass_pdf.png'.format(star.name,star.name))
    plt.close()
    
    
def read_star_profile(spfn):
    #Reads a text file with various info on the star and stores it in an object of the star_profile class
    #sp_fn - star profile filename
    
    prof = dict()
    
    #Read the profile and make a dictionary with all the profile values
    with open(spfn, 'r') as fn:
        fn_reader = csv.reader(fn, delimiter=' ')
        for line in fn_reader:
            if len(line) == 2:
                prof[line[0]] = line[1]
            elif len(line) == 3:
                prof[line[0]] = line[1:3]
            
    #Make the object of star_profile class
    star = star_profile(prof['name'],prof['evo'],prof['rad'],prof['rad_err'],prof['teff'],prof['teff_err'],prof['lum'],prof['lum_err'],prof['mass_range'],prof['mass_res'],prof['hist_mass_res'],prof['age_range'],prof['age_res'],prof['hist_age_res'])
    return star

def stringify_metallicity(met):
    #Takes in a float metallicity and returns a string of appropriate format
    if met >= 0:
        met_str = '+'
    else:
        met_str = ''
    met_str += '{:.2f}'.format(met)
    return met_str

def interpolate_mt(mt,age_range,age_res):
    l = mt.eeps['log_L']
    t = mt.eeps['log_Teff']
    a = mt.eeps['star_age']
    
    max_age = age_range[1]*1e6
    if max(a) < max_age:
        max_age = max(a)
    min_age = age_range[0]*1e6
    if min(a) > min_age:
        min_age = min(a)
    age_res *= 1e6
    
    lint = make_interp_spline(a,l)
    tint = make_interp_spline(a,t)
    ages = np.arange(min_age,max_age+age_res,age_res)
    
    mt_logt = [tint(a) for a in ages]
    mt_logl = [lint(a) for a in ages]
    
    return mt_logt,mt_logl,ages
    

def get_age_array(track_df):
    #Check the age arrays for the individual mass tracks, and get the smallest one to use for the group
    masses = track_df.mass.unique()
    first = True
    for m in masses:
        this_df = track_df[track_df.mass == m]
        if first:
            age_arr = np.array(this_df.age)
            first = False
        elif len(this_df.age) < len(age_arr):
            age_arr = np.array(this_df.age)
    return age_arr


def get_sig(track_col,val,valerrlo,valerrhi):
    #Get how many sigmas each point on the track_df is from the value
    sigs = [get_ind_sig(track_val,val,valerrlo,valerrhi) for track_val in track_col]
    return sigs
def get_ind_sig(track_val,val,valerrlo,valerrhi):
    if track_val >= val:
        sig = (track_val-val)/valerrhi
    else:
        sig = (val-track_val)/valerrlo
    return sig

def get_gauss(sig):
    gauss = math.exp(-sig**2/2)
    return gauss

def make_report(df,tp):
    #Takes a dataframe and writes a report for the age or mass
    #df - the dataframe 
    #tp - the type (age or mass)
    
    report = ''
    
    if tp == 'age':
        sl = df.age
        unit = 'Myr'
        sl = sl/1e6
    elif tp == 'mass':
        sl = df.star_mass
        unit = 'M_sun'
    
    n_entries = len(df)
    total_weight = sum(df.gauss)
    
    
    med   = np.percentile(sl,50,weights=df.gauss,method='inverted_cdf')
    sig1 = [np.percentile(sl,100*stats.norm.cdf(-1),weights=df.gauss,method='inverted_cdf'),np.percentile(sl,100*stats.norm.cdf(1),weights=df.gauss,method='inverted_cdf')]
    sig2 = [np.percentile(sl,100*stats.norm.cdf(-2),weights=df.gauss,method='inverted_cdf'),np.percentile(sl,100*stats.norm.cdf(2),weights=df.gauss,method='inverted_cdf')]
    
    report += 'Number of entries: {}\n'.format(n_entries)
    report += 'Total weight of entries: {}\n'.format(total_weight)
    report += 'Median: {} {}\n'.format(med,unit)
    report += '1 sig CI: {} {} - {} {}\n'.format(sig1[0],unit,sig1[1],unit)
    report += '2 sig CI: {} {} - {} {}\n'.format(sig2[0],unit,sig2[1],unit)
    
    return report,med,sig1,sig2

def build_iso_df(iso_files,mass_range,mass_res,age_range):
    #Build the dataframe with interpolate logt, logl, mass for each age
    first_iso = True
    
    plot_xs = []
    plot_ys = []
    plot_as = []
    
    for isofn in iso_files:
        iso = rmm.ISO(isofn)
        
        for i,a in enumerate(iso.ages):
            l = iso.isos[i]['log_L']
            t = iso.isos[i]['log_Teff']
            m_s = iso.isos[i]['star_mass']
            m_i = iso.isos[i]['initial_mass']
            
            #plt.plot(t,l,'k-')
            plot_xs.append(t)
            plot_ys.append(l)
            plot_as.append(10**a/1e6)   #This assumes the version 2.5 MIST isochrones, which lable age as log(age). This presents age as linear Myr.
            
            max_mass = mass_range[1]
            if max(m_i) < max_mass:
                max_mass = max(m_i)
            min_mass = mass_range[0]
            if min(m_i) > min_mass:
                min_mass = min(m_i)
            
            l,t,m_s,m_i = clean_for_iso(l,t,m_s,m_i) #in case there are any duplicate masses
            
            lint = make_interp_spline(m_i,l)
            tint = make_interp_spline(m_i,t)
            mint = make_interp_spline(m_i,m_s)
            masses = np.arange(min_mass,max_mass+mass_res,mass_res)
            
            iso_logt = [tint(m) for m in masses]
            iso_logl = [lint(m) for m in masses]
            iso_m_curr = [mint(m) for m in masses]
            
            data={'logt':iso_logt,'logl':iso_logl,'star_mass':iso_m_curr,'initial_mass':masses}
            this_df = pd.DataFrame(data=data)
            this_df['age'] = a
            if first_iso:
                iso_df = this_df
                first_iso = False
            else:
                iso_df = pd.concat([iso_df,this_df],ignore_index=True,axis=0)
                
    #Plot the isochrones
    lc = multiline(plot_xs,plot_ys,plot_as,cmap='gist_rainbow')
    cbar = plt.colorbar(lc)
    cbar.set_label('Isochrone Age (Myr)')
    
    return iso_df

def define_hist_age_bins(age_range,bin_width):
    #bin_width is the width of the bin in Myr
    a = age_range[0]
    age_bin = [a]
    a += bin_width
    while a <= age_range[1]:
        age_bin.append(a)
        a += bin_width
    return age_bin

def define_hist_mass_bins(mass_range,bin_width):
    #bin_width is the width of the bin in M_sun
    m = mass_range[0]
    mass_bin = [m]
    m += bin_width
    while m <= mass_range[1]:
        mass_bin.append(m)
        m += bin_width
    return mass_bin
    
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings
    Taken from: https://stackoverflow.com/a/50029441
    
    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection
    
    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)
    
    Returns
    -------
    lc : LineCollection instance.
    """
    
    # find axes
    ax = plt.gca() if ax is None else ax
    
    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)
    
    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))
    
    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    #ax.autoscale()
    return lc

def limit_between(target,limit,limit_range):
    #Limit the values of target such that the values of limit are within the limit_range
    #This assumes that limit is monotonic
    
    new_target = target[limit >= limit_range[0]]
    new_limit = limit[limit >= limit_range[0]]
    
    new_target = new_target[new_limit <= limit_range[1]]
    new_limit = new_limit[new_limit <= limit_range[1]]
    
    return new_target
    
def clean_for_iso(l,t,m_s,m_i):
    #print(type(l),type(t),type(m_s),type(m_i))
    bad_indicies = []
    for i,m in enumerate(m_i):
        if i!=0:
            if m == m_i[i-1]:
                bad_indicies.append(i)
    bad_indicies = np.array(bad_indicies)
    #print('Bad indicies: ',bad_indicies)
    if len(bad_indicies > 0):
        l = np.delete(l,bad_indicies)
        t = np.delete(t,bad_indicies)
        m_s = np.delete(m_s,bad_indicies)
        m_i = np.delete(m_i,bad_indicies)
    
    return l,t,m_s,m_i

class star_profile:
    def __init__(self,name,evo,r,rerr,t,terr,l,lerr,mass_range,mass_res,hist_mass_res,age_range,age_res,hist_age_res):
        #R & L in solar units, T in K
        self.name = name
        self.evo = evo
        self.r = float(r)
        self.rerr = float(rerr)
        self.t = float(t)
        self.terr = float(terr)
        self.l = float(l)
        self.lerr = float(lerr)
        self.logr = math.log10(self.r)
        self.logrerrhi = math.log10((self.r+self.rerr)/self.r)
        self.logrerrlo = math.log10(self.r/(self.r-self.rerr))
        self.logl = math.log10(self.l)
        self.loglerrhi = math.log10((self.l+self.lerr)/self.l)
        self.loglerrlo = math.log10(self.l/(self.l-self.lerr))
        self.logt = math.log10(self.t)
        self.logterrhi = math.log10((self.t+self.terr)/self.t)
        self.logterrlo = math.log10(self.t/(self.t-self.terr))
        self.mass_range = [float(mass_range[0]),float(mass_range[1])]
        self.mass_res = float(mass_res)
        self.hist_mass_res = float(hist_mass_res)
        self.age_range = [float(age_range[0]),float(age_range[1])]
        self.age_res = float(age_res)
        self.hist_age_res = float(hist_age_res)
        


if __name__=="__main__":
    main()