//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffParamsLoader.cc
//---------------------------------------------------------------------------//
#include "CutoffParamsLoader.hh"

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

#include "GdmlGeometryMap.hh"
#include "MaterialParamsLoader.hh"
#include "ParticleParamsLoader.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with RootLoader.
 */
CutoffParamsLoader::CutoffParamsLoader(RootLoader& root_loader)
    : root_loader_(root_loader)
{
    CELER_ENSURE(root_loader);
}

//---------------------------------------------------------------------------//
/*!
 * Load CutoffParams data.
 */
const std::shared_ptr<const CutoffParams> CutoffParamsLoader::operator()()
{
    const auto           tfile = root_loader_.get();
    MaterialParamsLoader material_loader(root_loader_);
    ParticleParamsLoader particle_loader(root_loader_);

    CutoffParams::Input input;
    input.materials = material_loader();
    input.particles = particle_loader();

    CELER_ENSURE(input.materials);
    CELER_ENSURE(input.particles);

    // Load geometry tree to access cutoff data
    std::unique_ptr<TTree> tree_geometry(tfile->Get<TTree>("geometry"));
    CELER_ASSERT(tree_geometry);
    CELER_ASSERT(tree_geometry->GetEntries() == 1);

    // Load branch data
    GdmlGeometryMap  geometry;
    GdmlGeometryMap* geometry_ptr = &geometry;

    int err_code
        = tree_geometry->SetBranchAddress("GdmlGeometryMap", &geometry_ptr);
    CELER_ASSERT(err_code >= 0);
    tree_geometry->GetEntry(0);

    for (auto pid : range(ParticleId{input.particles->size()}))
    {
        CutoffParams::MaterialCutoffs m_cutoffs;
        const auto                    pdg   = input.particles->id_to_pdg(pid);
        const auto& geometry_matid_material = geometry.matid_to_material_map();

        for (auto matid : range(MaterialId{input.materials->size()}))
        {
            const auto& material
                = geometry_matid_material.find(matid.get())->second;
            const auto& iter = material.pdg_cutoff.find(pdg.get());

            ParticleCutoff p_cutoff;
            if (iter != material.pdg_cutoff.end())
            {
                // Is a particle type with assigned cutoff values
                p_cutoff.energy = units::MevEnergy{iter->second.energy};
                p_cutoff.range  = iter->second.range;
            }
            else
            {
                // Set cutoffs to zero
                p_cutoff.energy = units::MevEnergy{zero_quantity()};
                p_cutoff.range  = 0;
            }
            m_cutoffs.push_back(p_cutoff);
        }
        input.cutoffs.insert({pdg, m_cutoffs});
    }

    CutoffParams cutoffs(input);
    return std::make_shared<CutoffParams>(std::move(cutoffs));
}

//---------------------------------------------------------------------------//
} // namespace celeritas
