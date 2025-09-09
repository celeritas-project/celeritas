//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/CutoffParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <memory>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "CutoffData.hh"
#include "CutoffView.hh"
#include "PDGNumber.hh"

namespace celeritas
{
class ParticleParams;
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Management particle and material cutoffs.
 *
 * Geant4 provides accessors to its production cuts from its
 * \c G4MaterialCutsCouple class, which couples cutoff and material data.
 * During import, for simplicity, G4's production cuts are stored alongside
 * the material information, in \c ImportPhysMaterial . Since this is a direct
 * import, the cutoff map in \c ImportPhysMaterial stores only the cuts
 * available in Geant4, i.e. only values for gammas, electrons, positrons, and
 * protons.
 *
 * In Celeritas, particle cutoff is stored contiguously in a single vector
 * of size num_particles * num_materials, which stores all particle cutoffs
 * for all materials. During import, any particle that is not in Geant4's
 * list receives a zero cutoff value. This opens the possibility to expand
 * cutoffs in the future, when data is not imported anymore.
 *
 * The \c Input structure provides a failsafe mechanism to construct the
 * host/device data.
 *
 * Some processes (e.g. photoelectric effect, decay) can produce secondaries
 * below the production threshold, while others (e.g. bremsstrahlung,
 * ionization) use the production cut as their intrinsic limit. By default all
 * of these secondaries are transported, even if their energy is below the
 * threshold. If the \c apply_post_interaction option is enabled, any secondary
 * photon, electron, or positron with energy below the cutoff will be killed
 * (the flag will be ignored for other particle types).
 */
class CutoffParams final : public ParamsDataInterface<CutoffParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using MaterialCutoffs = std::vector<ParticleCutoff>;
    //!@}

    //! Input data to construct this class
    struct Input
    {
        SPConstParticles particles;
        SPConstMaterials materials;
        std::map<PDGNumber, MaterialCutoffs> cutoffs;
        bool apply_post_interaction{false};
    };

  public:
    // Construct with imported data
    static std::shared_ptr<CutoffParams>
    from_import(ImportData const& data,
                SPConstParticles particle_params,
                SPConstMaterials material_params);

    // Construct with cutoff input data
    explicit CutoffParams(Input const& input);

    // Access cutoffs on host
    inline CutoffView get(PhysMatId material) const;

    //! Access cutoff data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access cutoff data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<CutoffParamsData> data_;
    using HostValue = HostVal<CutoffParamsData>;

    //// HELPER FUNCTIONS ////

    // PDG numbers of particles with prodution cuts
    static std::vector<PDGNumber> const& pdg_numbers();
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access cutoffs on host.
 */
CutoffView CutoffParams::get(PhysMatId material) const
{
    CELER_EXPECT(material < this->host_ref().num_materials);
    return CutoffView(this->host_ref(), material);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
