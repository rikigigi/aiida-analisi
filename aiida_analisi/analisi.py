from aiida.engine import CalcJob
from aiida.common import exceptions
from aiida.parsers.parser import Parser
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob

import aiida.orm
from aiida.orm import Int, Float, Str, List, Dict, ArrayData, Bool, TrajectoryData, CalcJobNode
import json
import numpy as np
import six


def get_types_id_array(types_array):
    types={}
    typeid=0
    res=[]
    for t in types_array:
        if not t in types:
            types[t]=typeid
            typeid=typeid+1
        res.append(types[t])
    return np.array(res,dtype='int32')

def get_analisi_traj_from_aiida(traj):
    import pyanalisi
    pos=traj.get_array('positions')
    vel=traj.get_array('velocities')
    cel=traj.get_array('cells')
    types=get_types_id_array(traj.get_attribute('symbols'))
    params= [pos, vel, types,  cel]
    atraj=pyanalisi.Trajectory(*params,True, False)
    return atraj


class AnalisiCalculation(CalcJob):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('traj', required=False, valid_type=(TrajectoryData), help='Input trajectory. It expects positions, cells and velocities arrays' )
        spec.input('n_blocks', default=lambda : Int(12), valid_type=(Int))
        spec.input('max_time', valid_type=(Int),default=lambda: Int(0))
        spec.input('skip', valid_type=(Int),default=lambda : Int(1))
        spec.input('start_step', valid_type=(Int), default=lambda : Int(0))
        spec.input('stop_step', valid_type=(Int), default=lambda : Int(-1))
        spec.input('metadata.options.parser_name', valid_type=six.string_types, default='analisi.analyze')
        spec.input('metadata.options.input_filename', valid_type=str, default='aiida.bin')
        spec.input('metadata.options.output_filename', valid_type=str, default='aiida.out')
        spec.inputs['metadata']['options']['resources'].default = {'num_machines': 1, 'num_mpiprocs_per_machine': 1}
        spec.inputs['metadata']['options']['withmpi'].default=True
        spec.input('msd',valid_type=Bool,default=lambda : Bool(False))
        spec.input('gofrt', valid_type=Dict, required=False, help="{'min_r': 0.5, 'max_r': 3.0, 'n_bins': 100}")
        spec.input('sh', valid_type=Dict, required=False, help="{'min_r': 0.5, 'max_r': 3.0, 'n_bins': 4}")
        spec.output('msd',valid_type=ArrayData)
        spec.output('gofrt',valid_type=ArrayData)
        spec.output('sh',valid_type=ArrayData)
        spec.exit_code(400,'ERROR_NO_DATA','You must provide a trajectory data')
        spec.exit_code(401,'ERROR_TOO_CALCULATIONS_SPECIFIED','You must specify only one input between msd, gofrt and sh')
        spec.exit_code(300, 'ERROR_NO_RETRIEVED_FOLDER',
            message='The retrieved folder data node could not be accessed.')
        spec.exit_code(310, 'ERROR_READING_OUTPUT_FILE',
            message='The output file could not be read from the retrieved folder.')
        spec.exit_code(320, 'ERROR_INVALID_OUTPUT',
            message='The output file contains invalid output.')

    def prepare_for_submission(self,folder):
        #create box.npy, coord.npy, force.npy, energy.npy and stress.npy (if present)

        symlink=[]
        copy_list=[]
        numthreads=self.inputs['metadata']['options']['resources']['num_cores_per_mpiproc']
        cmdline_params=['-N',numthreads,'-l','/dev/null', '-i', self.options.input_filename, '-B', str(self.inputs.n_blocks.value), '-S', str(self.inputs.max_time.value) , '-s', str(self.inputs.skip.value)]
        has_msd= 1 if self.inputs.msd.value else 0
        has_gofrt=1 if 'gofrt' in self.inputs else 0
        has_sh=1 if 'sh' in self.inputs else 0
        if has_msd+has_gofrt+has_sh != 1 :
             self.report('has_msd={} has_gofrt={} has_sh={}'.format(has_msd, has_gofrt, has_sh))
             self.report(self.inputs)
             raise self.exit_codes.ERROR_TOO_CALCULATIONS_SPECIFIED

        def get_p(d):
            return d['min_r'], d['max_r'],d['n_bins']

        if has_msd:
            cmdline_params=cmdline_params+['-Q']
        if has_gofrt:
            rmin,rmax,nbin=get_p(self.inputs.gofrt)
            cmdline_params=cmdline_params+['-g', str(nbin)]
        if has_sh:
            rmin,rmax,nbin=get_p(self.inputs.sh)
            cmdline_params=cmdline_params+['-Y', str(nbin)]
        if has_gofrt or has_sh:
            cmdline_params=cmdline_params+['-F', str(rmin),str(rmax) ]

        #write trajectory file
        pytraj=get_analisi_traj_from_aiida(self.inputs.traj)
        bin_name=folder.get_abs_path(self.options.input_filename)
        pytraj.write_lammps_binary(bin_name,0,-1)        

        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.cmdline_params =  cmdline_params
        codeinfo.withmpi = self.inputs.metadata.options.withmpi
        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = copy_list
        calcinfo.remote_symlink_list=symlink

        #or self.metadata.options.output_filename?
        #calcinfo.retrieve_list = [
        #            (self.options.output_filename,'.',1),
        #            (write_dict['save_ckpt']+'*','.',1),
        #            (write_dict['disp_file'],'.',1)
        #          ]
        calcinfo.retrieve_list = [
                    self.options.output_filename
                  ]


        return calcinfo

class AnalisiParser(Parser):
    def parse(self, **kwargs):
        try:
            output_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with output_folder.open(self.node.get_attribute('output_filename'),'r') as handle:
                has_msd= 1 if self.node.inputs.msd.value else 0
                has_gofrt=1 if 'gofrt' in self.node.inputs else 0
                has_sh=1 if 'sh' in self.node.inputs else 0
                if has_msd+has_gofrt+has_sh != 1 :
                    return self.exit_codes.ERROR_INVALID_OUTPUT
                arraydata=ArrayData()
                if has_sh==1 or has_msd==1:
                    result=np.loadtxt(handle)
                    arraydata.set_array('shcorr' if has_sh else 'msd',result)
                    arraydata.set_array('times',self.node.inputs.traj.get_array('times')[:result.shape[0]]-self.node.inputs.traj.get_array('times')[0])
                    self.out('sh' if has_sh else 'msd',arraydata)
                if has_gofrt==1:
                    lines=handle.readlines()
                    data=[]
                    block=''
                    br=True
                    for i,line in enumerate(lines):
                        if not line:
                            br=True
                        else:
                            if br:
                                data.append([])
                            br=False
                        data[-1].append(np.fromstring(line,sep=' '))
                    data=np.array(data)
                    arraydata.set_array('gofrt',data)
                    arraydata.set_array('times',self.node.inputs.traj.get_array('times')[:data.shape[0]])
                    self.out('gofrt',arraydata)
        except (OSError, IOError):
            return self.exit_codes.ERROR_READING_OUTPUT_FILE



