//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/MaterialTestBase.hh
//---------------------------------------------------------------------------//
#include "celeritas/Quantities.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class MaterialTestBase
{
  protected:
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;

    SPConstMaterial build_material()
    {
        using units::AmuMass;
        using units::MevMass;

        MaterialParams::Input inp;

        // Using nuclear masses provided by Geant4 11.0.3
        inp.isotopes = {
            // H
            {AtomicNumber{1}, AtomicNumber{1}, MevMass{938.272}, "1H"},
            {AtomicNumber{1}, AtomicNumber{2}, MevMass{1875.61}, "2H"},
            // Al
            {AtomicNumber{13}, AtomicNumber{27}, MevMass{25126.5}, "27Al"},
            {AtomicNumber{13}, AtomicNumber{28}, MevMass{26058.3}, "28Al"},
            // Na
            {AtomicNumber{11}, AtomicNumber{23}, MevMass{21409.2}, "23Na"},
            // I
            {AtomicNumber{53}, AtomicNumber{125}, MevMass{116321}, "125I"},
            {AtomicNumber{53}, AtomicNumber{126}, MevMass{117253}, "126I"},
            {AtomicNumber{53}, AtomicNumber{127}, MevMass{118184}, "127I"}};

        inp.elements = {
            // H
            {AtomicNumber{1},
             AmuMass{1.008},
             {{IsotopeId{0}, 0.9}, {IsotopeId{1}, 0.1}},
             "H"},
            // Al
            {AtomicNumber{13},
             AmuMass{26.9815385},
             {{IsotopeId{2}, 0.7}, {IsotopeId{3}, 0.3}},
             "Al"},
            // Na
            {AtomicNumber{11}, AmuMass{22.98976928}, {{IsotopeId{4}, 1}}, "Na"},
            // I
            {AtomicNumber{53},
             AmuMass{126.90447},
             {{IsotopeId{5}, 0.05}, {IsotopeId{6}, 0.15}, {IsotopeId{7}, 0.8}},
             "I"},
        };

        inp.materials = {
            // Sodium iodide
            {2.948915064677e+22,
             293.0,
             MatterState::solid,
             {{ElementId{2}, 0.5}, {ElementId{3}, 0.5}},
             "NaI"},
            // Void
            {0, 0, MatterState::unspecified, {}, "hard vacuum"},
            // Diatomic hydrogen
            {1.0739484359044669e+20,
             100.0,
             MatterState::gas,
             {{ElementId{0}, 1.0}},
             Label{"H2", "1"}},
            // Diatomic hydrogen with the same name and different properties
            {1.072e+20,
             110.0,
             MatterState::gas,
             {{ElementId{0}, 1.0}},
             Label{"H2", "2"}},
        };

        return std::make_shared<MaterialParams const>(std::move(inp));
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
