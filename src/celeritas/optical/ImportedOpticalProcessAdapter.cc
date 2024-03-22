//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedOpticalProcessAdapter.cc
//---------------------------------------------------------------------------//
#include "ImportedOpticalProcessAdapter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
IOPAContextException::IOPAContextException(ImportProcessClass ipc, MaterialId mid)
{
    std::stringstream os;
    os << "Optical process '" << to_cstring(ipc) << ", material ID=" << mid.unchecked_get();
    what_ = os.str();
}


std::shared_ptr<ImportedOpticalProcesses>
ImportedOpticalProcesses::from_import(ImportData const& data)
{
    CELER_EXPECT(std::all_of(
                data.optical_processes.begin(),
                data.optical_processes.end(),
                [](ImportOpticalProcess const& ip) { return bool(ip); }));

    ImportOpticalProcess absorption;
    absorption.process_class = ImportProcessClass::absorption;
    absorption.lamba_table.type = ImportTableType::lambda;

    ImportOpticalProcess rayleigh;
    rayleigh.process_class = ImportProcessClass::rayleigh;
    rayleigh.lambda_table.type = ImportTableType::lambda;

    for (auto mat_idx : range(ImportData::MatIdx{data.materials.size()}))
    {
        ImportMaterial const& material = data.materials[mat_idx];
        ImportOpticalMaterial const& optical_material = data.optical[mat_idx];

        absorption.physics_vectors.push_back(optical_material.absorption.absorption_length);
        rayleigh.physics_vectors.push_back(optical_material.rayleight.mfp);
    }

    return std::make_shared<ImportedOpticalProcesses>({std::move(absorption), std::move(rayleigh)});
}

ImportedOpticalProcesses::ImportedOpticalProcesses(std::vector<ImportOpticalProcess> io)
    : processes_(std::move(io))
{
    for (auto id : range(ImportOpticalProcessId{this->size()}))
    {
        ImportOpticalProcess const& ip = processes_[id.get()];

        auto insertion = ids_.insert({key_type{ip.process_class}, id});
        CELER_VALIDATE(insertion.second,
                << "encountered duplicate import opticla process class '"
                << to_cstring(ip.process_class) << "' for optical photons "
                << "(there may be at most one optical process of a given type)");
    }

    CELER_ENSURE(processes_.size() == ids_.size());
}

auto ImportedOpticalProcess::find(key_type process) const -> ImportedOpticalProcessId
{
    auto iter = ids_.find(process);
    if (iter == ids_.end())
        return {};
    return iter->second;
}

ImportedOpticalProcessAdapter::ImportedOpticalProcessAdapter(
        SPConstImported imported,
        ImportProcessClass process_class)
    : imported_(std::move(imported)), process_class_(process_class)
{
    CELER_ASSERT(imported_->find(process_class_),
            << "imported optical process data is unavailable (needed for '"
            << to_cstring(process_class_) << "')");
    CELER_EXPECT(this->get_lambda());
}

std::vector<OpticalValueGridId> ImportedOpticalProcessAdapter::step_limits(GenericGridInserter& inserter, MaterialParams const& mats) const
{
    try
    {
        std::vector<OpticalValueGridId> grid_ids;

        for (auto material : range(MaterialId{mats.size()}))
        {
            grid_ids.push_back(inserter(get_lambda().physics_vectors[material.get()]));
        }
        CELER_ASSERT(grid_ids.size() == mats.size());

        return grid_ids;
    }
    catch(...)
    {
        std::throw_with_nested(IOPAContextException(process_class, material));
    }
}


//---------------------------------------------------------------------------//
}  // namespace celeritas
