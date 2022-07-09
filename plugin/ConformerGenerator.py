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
        self.selected_complex_index = None
        self.max_conformers = 100
        self.sort_by = 'minimized energy'
        self.create_menu()
        self.update_structure_list()
        self.on_run()

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

        self.dd_conformer_count: ui.Dropdown = root.find_node('Dropdown Conformer Count').get_content()
        self.dd_conformer_count.register_item_clicked_callback(self.select_conformer_count)

        self.dd_sort_by: ui.Dropdown = root.find_node('Dropdown Sort By').get_content()
        self.dd_sort_by.register_item_clicked_callback(self.select_sort_by)

        self.btn_generate: ui.Button = root.find_node('Button Generate').get_content()
        self.btn_generate.register_pressed_callback(self.generate_conformers)
        self.btn_generate.disable_on_press = True

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

    def select_entry(self, dd: ui.Dropdown, ddi: ui.DropdownItem):
        self.update_content(dd)
        self.selected_complex_index = ddi.index

    def select_conformer_count(self, dd: ui.Dropdown, ddi: ui.DropdownItem):
        self.update_content(dd)
        self.max_conformers = int(ddi.name)

    def select_sort_by(self, dd: ui.Dropdown, ddi: ui.DropdownItem):
        self.update_content(dd)
        self.sort_by = ddi.name

    @async_callback
    async def generate_conformers(self, btn=None):
        temp_dir = tempfile.TemporaryDirectory()
        input_sdf = tempfile.NamedTemporaryFile(dir=temp_dir.name, delete=False, suffix='.sdf')
        output_sdf = tempfile.NamedTemporaryFile(dir=temp_dir.name, delete=False, suffix='.sdf')

        [complex] = await self.request_complexes([self.selected_complex_index])
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
        ref_mol = Chem.RemoveHs(mol)

        mol = Chem.AddHs(mol)
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=3*self.max_conformers)

        energies = []
        for cid in cids:
            AllChem.UFFOptimizeMolecule(mol, confId=cid)
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            energies.append(ff.CalcEnergy())

        mol = Chem.RemoveHs(mol)
        sorted_cids = sorted(cids, key=lambda id: energies[id])
        writer = Chem.SDWriter(output_sdf.name)

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

        sort_key = 1 if self.sort_by == 'minimized energy' else 2
        sorted_conformers = sorted(kept_confs_and_data, key=lambda x: x[sort_key])

        for cid, _, _ in sorted_conformers:
            writer.write(mol, cid)
        writer.close()

        new_complex = Complex.io.from_sdf(path=output_sdf.name)
        new_complex = new_complex.convert_to_frames()

        for i, new_m in enumerate(new_complex.molecules):
            new_m.associated['conf energy'] = str(sorted_conformers[i][1])
            new_m.associated['conf rmsd'] = str(sorted_conformers[i][2])
            new_m.associated.update(old_data)

        new_complex.current_frame = 0
        new_complex.name = complex.name
        new_complex.position = complex.position
        new_complex.rotation = complex.rotation

        self.update_content(self.btn_generate)
        await self.add_to_workspace([new_complex])

        # add_to_workspace has last frame selected, set it to first
        complexes = await self.request_complex_list()
        complexes[-1].current_frame = 0
        self.update_structures_shallow([complexes[-1]])


def main():
    description = 'A Nanome Plugin to generate conformers for a selected molecule using RDKit'
    plugin = nanome.Plugin('Conformer Generator', description, 'Tools', False)
    plugin.set_plugin_class(ConformerGenerator)
    plugin.run()


if __name__ == '__main__':
    main()
