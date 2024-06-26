//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/NeutronTestBase.cc
//---------------------------------------------------------------------------//
#include "NeutronTestBase.hh"

#include "corecel/math/ArrayUtils.hh"
#include "celeritas/mat/MaterialTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "TestMacros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Initialize particle and material params
 */
NeutronTestBase::NeutronTestBase()
{
    using namespace constants;
    using namespace units;
    constexpr auto zero = zero_quantity();

    constexpr MevMass neutron_mass{939.5654133};
    constexpr MevMass proton_mass{938.272013};

    // Setup default particle params
    ParticleParams::Input par_inp = {
        {"neutron", pdg::neutron(), neutron_mass, zero, stable_decay_constant},
        {"proton", pdg::proton(), proton_mass, zero, stable_decay_constant}};
    this->set_particle_params(std::move(par_inp));

    // Setup default material params
    MaterialParams::Input mat_inp;

    // Isotopes
    mat_inp.isotopes = {{AtomicNumber{2},
                         AtomicNumber{3},
                         units::MevEnergy{7.71804},
                         units::MevEnergy{5.49},
                         units::MevEnergy{44},
                         units::MevMass{3016.0},
                         "3He"},
                        {AtomicNumber{2},
                         AtomicNumber{4},
                         units::MevEnergy{28.2957},
                         units::MevEnergy{19.814},
                         units::MevEnergy{20.578},
                         units::MevMass{4002.6},
                         "4He"},
                        {AtomicNumber{29},
                         AtomicNumber{63},
                         units::MevEnergy{551.384},
                         units::MevEnergy{6.122},
                         units::MevEnergy{10.864},
                         units::MevMass{58618.5},
                         "63Cu"},
                        {AtomicNumber{29},
                         AtomicNumber{65},
                         units::MevEnergy{569.211},
                         units::MevEnergy{7.454},
                         units::MevEnergy{9.911},
                         units::MevMass{60479.8},
                         "65Cu"},
                        {AtomicNumber{82},
                         AtomicNumber{202},
                         units::MevEnergy{1607.5056},
                         units::MevEnergy{6.637},
                         units::MevEnergy{8.395},
                         units::MevMass{189958.3388},
                         "202Pb"},
                        {AtomicNumber{82},
                         AtomicNumber{206},
                         units::MevEnergy{1622.3240},
                         units::MevEnergy{7.254},
                         units::MevEnergy{8.087},
                         units::MevMass{191822.6512},
                         "206Pb"},
                        {AtomicNumber{82},
                         AtomicNumber{207},
                         units::MevEnergy{1629.0618},
                         units::MevEnergy{7.488},
                         units::MevEnergy{6.738},
                         units::MevMass{192755.4787},
                         "207Pb"},
                        {AtomicNumber{82},
                         AtomicNumber{208},
                         units::MevEnergy{1636.4296},
                         units::MevEnergy{8.004},
                         units::MevEnergy{7.368},
                         units::MevMass{193687.6762},
                         "208Pb"},
                        {AtomicNumber{5},
                         AtomicNumber{10},
                         units::MevEnergy{64.751},
                         units::MevEnergy{6.587},
                         units::MevEnergy{8.437},
                         units::MevMass{9324.437},
                         "10B"},
                        {AtomicNumber{5},
                         AtomicNumber{11},
                         units::MevEnergy{76.205},
                         units::MevEnergy{11.229},
                         units::MevEnergy{11.454},
                         units::MevMass{10252.548},
                         "11B"}};

    // Elements
    mat_inp.elements = {{AtomicNumber{2},
                         units::AmuMass{4.0026},
                         {{IsotopeId{0}, 0.001}, {IsotopeId{1}, 0.999}},
                         Label{"He"}},
                        {AtomicNumber{29},
                         units::AmuMass{63.546},
                         {{IsotopeId{2}, 0.692}, {IsotopeId{3}, 0.308}},
                         Label{"Cu"}}};
    // Materials
    mat_inp.materials = {{native_value_from(MolCcDensity{1e-2}),
                          293.2,
                          MatterState::liquid,
                          {{ElementId{0}, 1.0}},
                          Label{"He"}},
                         {native_value_from(MolCcDensity{0.141}),
                          293.2,
                          MatterState::solid,
                          {{ElementId{1}, 1.0}},
                          Label{"Cu"}},
                         {native_value_from(MolCcDensity{0.128}),
                          293.2,
                          MatterState::solid,  // just for the purpose of
                                               // testing
                          {{ElementId{0}, 0.10}, {ElementId{1}, 0.90}},
                          Label{"HeCu"}}};
    this->set_material_params(std::move(mat_inp));
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor.
 */
NeutronTestBase::~NeutronTestBase() = default;

//---------------------------------------------------------------------------//
/*!
 * Set particle parameters.
 */
void NeutronTestBase::set_material_params(MaterialParams::Input inp)
{
    CELER_EXPECT(!inp.materials.empty());

    material_params_ = std::make_shared<MaterialParams>(std::move(inp));
    ms_ = StateStore<MaterialStateData>(material_params_->host_ref(), 1);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the incident track's material
 */
void NeutronTestBase::set_material(std::string const& name)
{
    CELER_EXPECT(material_params_);

    mt_view_ = std::make_shared<MaterialTrackView>(
        material_params_->host_ref(), ms_.ref(), TrackSlotId{0});

    // Initialize
    MaterialTrackView::Initializer_t init;
    init.material_id = material_params_->find_material(name);
    CELER_VALIDATE(init.material_id, << "no material '" << name << "' exists");
    *mt_view_ = init;
}

//---------------------------------------------------------------------------//
/*!
 * Set particle parameters.
 */
void NeutronTestBase::set_particle_params(ParticleParams::Input inp)
{
    CELER_EXPECT(!inp.empty());
    particle_params_ = std::make_shared<ParticleParams>(std::move(inp));
    ps_ = StateStore<ParticleStateData>(particle_params_->host_ref(), 1);
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the incident particle data
 */
void NeutronTestBase::set_inc_particle(PDGNumber pdg, MevEnergy energy)
{
    CELER_EXPECT(particle_params_);
    CELER_EXPECT(pdg);
    CELER_EXPECT(energy >= zero_quantity());

    // Construct track view
    pt_view_ = std::make_shared<ParticleTrackView>(
        particle_params_->host_ref(), ps_.ref(), TrackSlotId{0});

    // Initialize
    ParticleTrackView::Initializer_t init;
    init.particle_id = particle_params_->find(pdg);
    init.energy = energy;
    *pt_view_ = init;
}

//---------------------------------------------------------------------------//
/*!
 * Set an incident direction (and normalize it).
 */
void NeutronTestBase::set_inc_direction(Real3 const& dir)
{
    CELER_EXPECT(norm(dir) > 0);

    inc_direction_ = make_unit_vector(dir);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
