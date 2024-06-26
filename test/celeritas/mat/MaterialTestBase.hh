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
        using namespace celeritas::units;

        MaterialParams::Input inp;

        // Using nuclear masses provided by Geant4 11.0.3
        inp.isotopes = {// H
                        {AtomicNumber{1},
                         AtomicNumber{1},
                         MevEnergy{0},
                         MevEnergy{0},
                         MevEnergy{0},
                         MevMass{938.272},
                         "1H"},
                        {AtomicNumber{1},
                         AtomicNumber{2},
                         MevEnergy{2.22457},
                         MevEnergy{2.22457},
                         MevEnergy{2.22457},
                         MevMass{1875.61},
                         "2H"},
                        // Al
                        {AtomicNumber{13},
                         AtomicNumber{27},
                         MevEnergy{224.952},
                         MevEnergy{8.271},
                         MevEnergy{13.058},
                         MevMass{25126.5},
                         "27Al"},
                        {AtomicNumber{13},
                         AtomicNumber{28},
                         MevEnergy{232.677},
                         MevEnergy{9.553},
                         MevEnergy{7.725},
                         MevMass{26058.3},
                         "28Al"},
                        // Na
                        {AtomicNumber{11},
                         AtomicNumber{23},
                         MevEnergy{186.564},
                         MevEnergy{8.794},
                         MevEnergy{12.420},
                         MevMass{21409.2},
                         "23Na"},
                        // I
                        {AtomicNumber{53},
                         AtomicNumber{125},
                         MevEnergy{1056.29},
                         MevEnergy{5.601},
                         MevEnergy{9.543},
                         MevMass{116321},
                         "125I"},
                        {AtomicNumber{53},
                         AtomicNumber{126},
                         MevEnergy{1063.43},
                         MevEnergy{6.177},
                         MevEnergy{7.145},
                         MevMass{117253},
                         "126I"},
                        {AtomicNumber{53},
                         AtomicNumber{127},
                         MevEnergy{1072.58},
                         MevEnergy{6.208},
                         MevEnergy{9.144},
                         MevMass{118184},
                         "127I"}};

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
            {native_value_from(InvCcDensity{2.948915064677e+22}),
             293.0,
             MatterState::solid,
             {{ElementId{2}, 0.5}, {ElementId{3}, 0.5}},
             "NaI"},
            // Void
            {0, 0, MatterState::unspecified, {}, "hard vacuum"},
            // Diatomic hydrogen
            {native_value_from(InvCcDensity{1.0739484359044669e+20}),
             100.0,
             MatterState::gas,
             {{ElementId{0}, 1.0}},
             Label{"H2", "1"}},
            // Diatomic hydrogen with the same name and different properties
            {native_value_from(InvCcDensity{1.072e+20}),
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
