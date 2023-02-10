import os
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.rdMolDescriptors as mDesc

import nanome
from nanome import ui
from nanome.api.structure import Complex
from nanome.util import async_callback, Logs
from nanome.util.enums import NotificationTypes

BASE_DIR = os.path.dirname(__file__)
MENU_PATH = os.path.join(BASE_DIR, 'menu.json')
MAX_ROTATABLE_BONDS = 20


class ConformerGenerator(nanome.AsyncPluginInstance):
    def start(self):
        self.set_plugin_list_button(self.PluginListButtonType.run, 'Open')

        self.selected_complex_index = None
        self.selected_ligand_index = None
        self.selected_complex = None
        self.selected_ligand = None

        self.max_conformers = 100
        self.sort_by = 'minimized energy'
        self.lock_results = False

        self.create_menu()
        self.update_structure_list()

    def on_run(self):
        self.menu.enabled = True
        self.update_menu(self.menu)

    def on_complex_list_changed(self):
        self.update_structure_list()

    def create_menu(self):
        self.menu = ui.Menu.io.from_json(MENU_PATH)
        root: ui.LayoutNode = self.menu.root

        self.dd_entry: ui.Dropdown = root.find_node('Dropdown Entry').get_content()
        self.dd_entry.register_item_clicked_callback(self.select_entry)

        self.ln_dd_ligand: ui.LayoutNode = root.find_node('Dropdown Ligand')
        self.dd_ligand: ui.Dropdown = self.ln_dd_ligand.get_content()
        self.dd_ligand.register_item_clicked_callback(self.select_ligand)

        self.dd_conformer_count: ui.Dropdown = root.find_node('Dropdown Conformer Count').get_content()
        self.dd_conformer_count.register_item_clicked_callback(self.select_conformer_count)

        self.dd_sort_by: ui.Dropdown = root.find_node('Dropdown Sort By').get_content()
        self.dd_sort_by.register_item_clicked_callback(self.select_sort_by)

        ln_toggle_lock: ui.LayoutNode = root.find_node('Toggle Lock')
        self.btn_toggle_lock = ln_toggle_lock.add_new_toggle_switch('Lock entry and result')
        self.btn_toggle_lock.register_pressed_callback(self.toggle_lock)
        self.btn_toggle_lock.enabled = self.lock_results

        self.btn_generate: ui.Button = root.find_node('Button Generate').get_content()
        self.btn_generate.register_pressed_callback(self.generate_conformers)
        self.btn_generate.disable_on_press = True

    def reset_selection(self):
        self.selected_complex = None
        self.selected_complex_index = None
        self.selected_ligand = None
        self.selected_ligand_index = None

    def hide_ligand_list(self):
        self.dd_ligand.items.clear()
        self.ln_dd_ligand.enabled = False
        self.update_node(self.ln_dd_ligand)

    @async_callback
    async def update_structure_list(self):
        complexes = await self.request_complex_list()
        self.dd_entry.items.clear()
        for complex in complexes:
            ddi = ui.DropdownItem(complex.full_name)
            ddi.index = complex.index
            if self.selected_complex_index == ddi.index:
                ddi.selected = True
            self.dd_entry.items.append(ddi)

        indices = [complex.index for complex in complexes]
        if self.selected_complex_index not in indices:
            self.selected_complex_index = None

        if self.selected_complex_index is None and complexes:
            self.dd_entry.items[0].selected = True
            self.select_entry(self.dd_entry, self.dd_entry.items[0])
        self.update_content(self.dd_entry)

    @async_callback
    async def update_ligand_list(self):
        if not self.selected_complex:
            self.hide_ligand_list()
            return

        molecule = next(
            mol for i, mol in enumerate(self.selected_complex.molecules)
            if i == self.selected_complex.current_frame)
        ligands = await molecule.get_ligands()

        if not ligands:
            self.hide_ligand_list()
            return

        ligand_complexes = []
        for ligand in ligands:
            ligand_complex = nanome.structure.Complex()
            ligand_complex.position = self.selected_complex.position
            ligand_complex.rotation = self.selected_complex.rotation
            ligand_complex.name = ligand.name
            ligand_molecule = nanome.structure.Molecule()
            ligand_chain = nanome.structure.Chain()
            ligand_complex.add_molecule(ligand_molecule)
            ligand_molecule.add_chain(ligand_chain)
            for residue in ligand.residues:
                ligand_chain.add_residue(residue)
            ligand_complexes.append(ligand_complex)

        if len(ligand_complexes) == 1:
            mol_num_atoms = sum(1 for _ in molecule.atoms)
            lig_num_atoms = sum(1 for _ in ligand_complexes[0].atoms)
            if mol_num_atoms == lig_num_atoms:
                self.hide_ligand_list()
                return

        self.dd_ligand.items.clear()
        for i, ligand in enumerate([None, *ligand_complexes]):
            ddi = ui.DropdownItem(ligand.name if ligand else 'None')
            ddi.index = i
            ddi.ligand = ligand
            if self.selected_ligand_index == ddi.index:
                ddi.selected = True
            self.dd_ligand.items.append(ddi)

        self.ln_dd_ligand.enabled = True
        self.update_node(self.ln_dd_ligand)

        if self.selected_ligand_index is None and ligand_complexes:
            select_index = self.selected_ligand_index or 1
            ddi = self.dd_ligand.items[select_index]
            ddi.selected = True
            self.select_ligand(self.dd_ligand, ddi)

    @async_callback
    async def select_entry(self, dd: ui.Dropdown, ddi: ui.DropdownItem):
        self.update_content(dd)
        self.reset_selection()
        self.selected_complex_index = ddi.index
        [complex] = await self.request_complexes([self.selected_complex_index])
        self.selected_complex = complex
        self.update_ligand_list()

    def select_ligand(self, dd: ui.Dropdown, ddi: ui.DropdownItem):
        self.update_content(dd)
        self.selected_ligand_index = ddi.index
        self.selected_ligand = ddi.ligand

    def select_conformer_count(self, dd: ui.Dropdown, ddi: ui.DropdownItem):
        self.update_content(dd)
        self.max_conformers = int(ddi.name)

    def select_sort_by(self, dd: ui.Dropdown, ddi: ui.DropdownItem):
        self.update_content(dd)
        self.sort_by = ddi.name

    def toggle_lock(self, btn: ui.Button):
        self.lock_results = btn.selected

    @async_callback
    async def generate_conformers(self, btn=None):
        temp_dir = tempfile.TemporaryDirectory()
        input_sdf = tempfile.NamedTemporaryFile(dir=temp_dir.name, delete=False, suffix='.sdf')
        output_sdf = tempfile.NamedTemporaryFile(dir=temp_dir.name, delete=False, suffix='.sdf')

        # get latest state of selected complex
        [self.selected_complex] = await self.request_complexes([self.selected_complex_index])
        complex = self.selected_ligand or self.selected_complex
        old_data = {}

        # get only the current frame
        if len(list(complex.molecules)) == 1:
            m = next(complex.molecules)
            m.move_conformer(m.current_conformer, 0)
            m.set_conformer_count(1)
            old_data = m.associateds[0]
        else:
            m = list(complex.molecules)[complex.current_frame]
            complex.molecules = []
            complex.add_molecule(m)
            old_data = m.associated

        complex.io.to_sdf(input_sdf.name)
        mol = Chem.SDMolSupplier(input_sdf.name)[0]
        if mol is None:
            self.send_notification(NotificationTypes.error, 'RDKit error loading entry')
            return

        if mDesc.CalcNumRotatableBonds(mol) > MAX_ROTATABLE_BONDS:
            msg = f'Selected entry has too many (>{MAX_ROTATABLE_BONDS}) rotatable bonds'
            self.send_notification(NotificationTypes.error, msg)
            return

        # save starting conformer
        ref_mol = Chem.AddHs(mol, addCoords=True)
        AllChem.UFFOptimizeMolecule(ref_mol)
        ff = AllChem.UFFGetMoleculeForceField(ref_mol)
        ref_energy = ff.CalcEnergy()
        ref_mol = Chem.RemoveHs(mol)

        mol = Chem.AddHs(mol, addCoords=True)
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=3*self.max_conformers)

        # calc energies of conformers
        energies = []
        for cid in cids:
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            energies.append(ff.CalcEnergy())

        # sort conformers by energy
        mol = Chem.RemoveHs(mol)
        sorted_cids = sorted(cids, key=lambda id: energies[id])
        writer = Chem.SDWriter(output_sdf.name)
        writer.SetForceV3000(True)

        # filter conformers, prioritizing lower energy and ignoring similar conformers
        kept_confs_and_data = []
        for cid in sorted_cids:
            if len(kept_confs_and_data) >= self.max_conformers:
                break
            keep = True
            for other_cid, _, _ in kept_confs_and_data:
                rmsd = AllChem.GetBestRMS(mol, mol, prbId=cid, refId=other_cid)
                if rmsd < 0.5:
                    keep = False
                    break
            if keep:
                rmsd = AllChem.GetBestRMS(mol, ref_mol, prbId=cid, refId=0)
                kept_confs_and_data.append((cid, energies[cid], rmsd))

        # sort by energy or rmsd
        sort_key = 1 if self.sort_by == 'minimized energy' else 2
        sorted_conformers = sorted(kept_confs_and_data, key=lambda x: x[sort_key])
        sorted_conformers.insert(0, (0, ref_energy, 0.0))

        # write conformers to sdf, original conformer first
        writer.write(ref_mol)
        for cid, _, _ in sorted_conformers[1:]:
            writer.write(mol, cid)
        writer.close()

        new_complex = Complex.io.from_sdf(path=output_sdf.name)
        new_complex = new_complex.convert_to_frames()

        # add energy and rmsd to associated data
        for i, new_m in enumerate(new_complex.molecules):
            new_m.associated.update(old_data)
            new_m.associated['conf energy'] = str(round(sorted_conformers[i][1], 3))
            new_m.associated['conf rmsd'] = str(round(sorted_conformers[i][2], 3))

        # add conformers to complex
        new_complex.name = f'{complex.name} conf'
        new_complex.current_frame = 0
        new_complex.position = complex.position
        new_complex.rotation = complex.rotation
        new_complex.locked = self.lock_results
        new_complex.boxed = False

        self.update_content(self.btn_generate)
        await self.add_to_workspace([new_complex])

        if self.lock_results:
            self.selected_complex.locked = True
            self.update_structures_shallow([self.selected_complex])
            # temp hack because locked enables boxed
            self.selected_complex.boxed = False
            self.update_structures_shallow([self.selected_complex])

        # add_to_workspace has last frame selected, set it to first
        complexes = await self.request_complex_list()
        complexes[-1].current_frame = 0
        self.update_structures_shallow([complexes[-1]])

        self.send_notification(NotificationTypes.success, 'Conformers generated, check entry list')


def main():
    description = 'A Nanome Plugin to generate conformers for a selected molecule using RDKit'
    plugin = nanome.Plugin('Conformer Generator', description, 'Tools', False)
    plugin.set_plugin_class(ConformerGenerator)
    plugin.run()


if __name__ == '__main__':
    main()
