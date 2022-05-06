//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/ParticleParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "corecel/data/CollectionMirror.hh"

#include "PDGNumber.hh"
#include "ParticleData.hh"
#include "ParticleView.hh"

namespace celeritas
{
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Data management for Standard Model particle classifications.
 *
 * This class represents "per-problem" shared data about standard
 * model particles being used.
 *
 * The ParticleParams is constructed on the host with a vector that
 * combines metadata (used for debugging output and interfacing with physics
 * setup) and data (used for on-device transport). Each entry in the
 * construction is assigned a unique \c ParticleId used for runtime access.
 *
 * The PDG Monte Carlo number is a unique "standard model" identifier for a
 * particle. See "Monte Carlo Particle Numbering Scheme" in the "Review of
 * Particle Physics":
 * https://pdg.lbl.gov/2020/reviews/rpp2020-rev-monte-carlo-numbering.pdf
 * It should be used to identify particle types during construction time.
 */
class ParticleParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef
        = ParticleParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = ParticleParamsData<Ownership::const_reference, MemSpace::device>;
    //!@}

    //! Define a particle's input data
    struct ParticleInput
    {
        std::string             name;     //!< Particle name
        PDGNumber               pdg_code; //!< See "Review of Particle Physics"
        units::MevMass          mass;     //!< Rest mass [MeV / c^2]
        units::ElementaryCharge charge;   //!< Charge in units of [e]
        real_type               decay_constant; //!< Decay constant [1/s]
    };

    //! Input data to construct this class
    using Input = std::vector<ParticleInput>;

  public:
    // Construct with imported data
    static std::shared_ptr<ParticleParams> from_import(const ImportData& data);

    // Construct with a vector of particle definitions
    explicit ParticleParams(const Input& defs);

    //// HOST ACCESSORS ////

    //! Number of particle definitions
    ParticleId::size_type size() const { return md_.size(); }

    // Get particle name
    inline const std::string& id_to_label(ParticleId id) const;

    // Get PDG code
    inline PDGNumber id_to_pdg(ParticleId id) const;

    // Find the ID from a name
    inline ParticleId find(const std::string& name) const;

    // Find the ID from a PDG code
    inline ParticleId find(PDGNumber pdg_code) const;

    // Access particle properties on host
    ParticleView get(ParticleId id) const;

    //! Access material properties on the host
    const HostRef& host_ref() const { return data_.host(); }

    //! Access material properties on the device
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    // Saved copy of metadata
    std::vector<std::pair<std::string, PDGNumber>> md_;

    // Map particle names to registered IDs
    std::unordered_map<std::string, ParticleId> name_to_id_;

    // Map particle codes to registered IDs
    std::unordered_map<PDGNumber, ParticleId> pdg_to_id_;

    // Host/device storage and reference
    CollectionMirror<ParticleParamsData> data_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get particle name.
 */
const std::string& ParticleParams::id_to_label(ParticleId id) const
{
    CELER_EXPECT(id < this->size());
    return md_[id.get()].first;
}

//---------------------------------------------------------------------------//
/*!
 * Get PDG code for a particle ID.
 */
PDGNumber ParticleParams::id_to_pdg(ParticleId id) const
{
    CELER_EXPECT(id < this->size());
    return md_[id.get()].second;
}

//---------------------------------------------------------------------------//
/*!
 * Find the ID from a name.
 */
ParticleId ParticleParams::find(const std::string& name) const
{
    auto iter = name_to_id_.find(name);
    if (iter == name_to_id_.end())
    {
        return ParticleId{};
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Find the ID from a PDG code.
 */
ParticleId ParticleParams::find(PDGNumber pdg_code) const
{
    auto iter = pdg_to_id_.find(pdg_code);
    if (iter == pdg_to_id_.end())
    {
        return ParticleId{};
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
