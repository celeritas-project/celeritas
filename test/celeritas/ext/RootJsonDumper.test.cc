//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootJsonDumper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/RootJsonDumper.hh"

#include <sstream>

#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportDataTrimmer.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class RootJsonDumperTest : public ::celeritas::test::Test
{
};

TEST_F(RootJsonDumperTest, all)
{
    // Import data
    ImportData imported = [this] {
        ScopedRootErrorHandler scoped_root_error;
        RootImporter import(
            this->test_data_path("celeritas", "four-steel-slabs.root"));
        auto result = import();
        scoped_root_error.throw_if_errors();
        return result;
    }();

    // Trim data
    {
        ImportDataTrimmer::Input inp;
        inp.materials = true;
        inp.physics = true;
        inp.mupp = true;
        inp.max_size = 2;
        ImportDataTrimmer trim{inp};
        trim(imported);
    }

    std::string str = [&imported] {
        std::ostringstream os;
        ScopedRootErrorHandler scoped_root_error;
        RootJsonDumper{os}(imported);
        scoped_root_error.throw_if_errors();
        return std::move(os).str();
    }();

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_JSON_EQ(
            R"json({
  "_typename" : "celeritas::ImportData",
  "isotopes" : [{
    "_typename" : "celeritas::ImportIsotope",
    "name" : "Fe54",
    "atomic_number" : 26,
    "atomic_mass_number" : 54,
    "binding_energy" : 471.76398226,
    "proton_loss_energy" : 8.85379258999995,
    "neutron_loss_energy" : 13.37845014,
    "nuclear_mass" : 50231.172508455
  }, {
    "_typename" : "celeritas::ImportIsotope",
    "name" : "H2",
    "atomic_number" : 1,
    "atomic_mass_number" : 2,
    "binding_energy" : 2.22456599,
    "proton_loss_energy" : 0,
    "neutron_loss_energy" : 0,
    "nuclear_mass" : 1875.6127932681
  }],
  "elements" : [{
    "_typename" : "celeritas::ImportElement",
    "name" : "Fe",
    "atomic_number" : 26,
    "atomic_mass" : 55.845110798,
    "isotopes_fractions" : [{
      "_typename" : "pair<unsigned int,double>",
      "first" : 0,
      "second" : 0.05845
    }, {
      "_typename" : "pair<unsigned int,double>",
      "first" : 3,
      "second" : 0.00282
    }]
  }, {
    "_typename" : "celeritas::ImportElement",
    "name" : "H",
    "atomic_number" : 1,
    "atomic_mass" : 1.00794075266514,
    "isotopes_fractions" : [{
      "_typename" : "pair<unsigned int,double>",
      "first" : 13,
      "second" : 0.999885
    }, {
      "_typename" : "pair<unsigned int,double>",
      "first" : 14,
      "second" : 1.15e-4
    }]
  }],
  "geo_materials" : [{
    "_typename" : "celeritas::ImportGeoMaterial",
    "name" : "G4_STAINLESS-STEEL",
    "state" : 1,
    "temperature" : 293.15,
    "number_density" : 86993489258991530803200,
    "elements" : [{
      "_typename" : "celeritas::ImportMatElemComponent",
      "element_id" : 0,
      "number_fraction" : 0.74
    }, {
      "_typename" : "celeritas::ImportMatElemComponent",
      "element_id" : 2,
      "number_fraction" : 0.0800000000000001
    }]
  }, {
    "_typename" : "celeritas::ImportGeoMaterial",
    "name" : "G4_Galactic",
    "state" : 3,
    "temperature" : 2.73,
    "number_density" : 0.0597469716754344,
    "elements" : [{
      "_typename" : "celeritas::ImportMatElemComponent",
      "element_id" : 3,
      "number_fraction" : 1
    }]
  }],
  "phys_materials" : [{
    "_typename" : "celeritas::ImportPhysMaterial",
    "geo_material_id" : 1,
    "optical_material_id" : 4294967295,
    "pdg_cutoffs" : [{"$pair" : "pair<int,celeritas::ImportProductionCut>", "first" : -11, "second" : {
      "_typename" : "celeritas::ImportProductionCut",
      "energy" : 9.9e-4,
      "range" : 0.1
    }}, {"$pair" : "pair<int,celeritas::ImportProductionCut>", "first" : 11, "second" : {
      "_typename" : "celeritas::ImportProductionCut",
      "energy" : 9.9e-4,
      "range" : 0.1
    }}, {"$pair" : "pair<int,celeritas::ImportProductionCut>", "first" : 22, "second" : {
      "_typename" : "celeritas::ImportProductionCut",
      "energy" : 9.9e-4,
      "range" : 0.1
    }}]
  }, {
    "_typename" : "celeritas::ImportPhysMaterial",
    "geo_material_id" : 0,
    "optical_material_id" : 4294967295,
    "pdg_cutoffs" : [{"$pair" : "pair<int,celeritas::ImportProductionCut>", "first" : -11, "second" : {
      "_typename" : "celeritas::ImportProductionCut",
      "energy" : 1.23589307919354,
      "range" : 0.1
    }}, {"$pair" : "pair<int,celeritas::ImportProductionCut>", "first" : 11, "second" : {
      "_typename" : "celeritas::ImportProductionCut",
      "energy" : 1.30827815530759,
      "range" : 0.1
    }}, {"$pair" : "pair<int,celeritas::ImportProductionCut>", "first" : 22, "second" : {
      "_typename" : "celeritas::ImportProductionCut",
      "energy" : 0.0208224420866223,
      "range" : 0.1
    }}]
  }],
  "regions" : [{
    "_typename" : "celeritas::ImportRegion",
    "name" : "DefaultRegionForTheWorld",
    "field_manager" : false,
    "production_cuts" : true,
    "user_limits" : false
  }, {
    "_typename" : "celeritas::ImportRegion",
    "name" : "DefaultRegionForParallelWorld",
    "field_manager" : false,
    "production_cuts" : true,
    "user_limits" : false
  }],
  "volumes" : [{
    "_typename" : "celeritas::ImportVolume",
    "geo_material_id" : 0,
    "region_id" : 0,
    "phys_material_id" : 1,
    "name" : "box@0",
    "solid_name" : "box"
  }, {
    "_typename" : "celeritas::ImportVolume",
    "geo_material_id" : 1,
    "region_id" : 0,
    "phys_material_id" : 0,
    "name" : "World",
    "solid_name" : "World"
  }],
  "particles" : [{
    "_typename" : "celeritas::ImportParticle",
    "name" : "e+",
    "pdg" : -11,
    "mass" : 0.51099891,
    "charge" : 1,
    "spin" : 0.5,
    "lifetime" : -1,
    "is_stable" : true
  }, {
    "_typename" : "celeritas::ImportParticle",
    "name" : "mu-",
    "pdg" : 13,
    "mass" : 105.6583715,
    "charge" : -1,
    "spin" : 0.5,
    "lifetime" : 2.19698e-6,
    "is_stable" : false
  }],
  "processes" : [{
    "_typename" : "celeritas::ImportProcess",
    "particle_pdg" : -11,
    "secondary_pdg" : 22,
    "process_type" : 2,
    "process_class" : 13,
    "models" : [{
      "_typename" : "celeritas::ImportModel",
      "model_class" : 13,
      "materials" : [{
        "_typename" : "celeritas::ImportModelMaterial",
        "energy" : [1e-4, 100000000],
        "micro_xs" : []
      }, {
        "_typename" : "celeritas::ImportModelMaterial",
        "energy" : [1e-4, 100000000],
        "micro_xs" : []
      }]
    }],
    "tables" : [],
    "applies_at_rest" : true
  }, {
    "_typename" : "celeritas::ImportProcess",
    "particle_pdg" : 13,
    "secondary_pdg" : 22,
    "process_type" : 2,
    "process_class" : 15,
    "models" : [{
      "_typename" : "celeritas::ImportModel",
      "model_class" : 20,
      "materials" : [{
        "_typename" : "celeritas::ImportModelMaterial",
        "energy" : [100, 100000000],
        "micro_xs" : [[1.44873278987692e-29, 3.86367883566152e-29]]
      }, {
        "_typename" : "celeritas::ImportModelMaterial",
        "energy" : [100, 100000000],
        "micro_xs" : [[3.00641640152192e-27, 8.8885205839549e-27], [3.46220315764337e-27, 1.02199532648639e-26]]
      }]
    }],
    "tables" : [{
      "_typename" : "celeritas::ImportPhysicsTable",
      "table_type" : 0,
      "x_units" : 1,
      "y_units" : 4,
      "physics_vectors" : [{
        "_typename" : "celeritas::ImportPhysicsVector",
        "vector_type" : 2,
        "x" : [100, 100000000],
        "y" : [0, 2.30843109957244e-30]
      }, {
        "_typename" : "celeritas::ImportPhysicsVector",
        "vector_type" : 2,
        "x" : [100, 100000000],
        "y" : [0, 7.63129425045843e-4]
      }]
    }],
    "applies_at_rest" : false
  }],
  "msc_models" : [{
    "_typename" : "celeritas::ImportMscModel",
    "particle_pdg" : -11,
    "model_class" : 3,
    "xs_table" : {
      "_typename" : "celeritas::ImportPhysicsTable",
      "table_type" : 4,
      "x_units" : 1,
      "y_units" : 6,
      "physics_vectors" : [{
        "_typename" : "celeritas::ImportPhysicsVector",
        "vector_type" : 2,
        "x" : [1e-4, 100],
        "y" : [3.64953143614647e-27, 1.39709799580588e-25]
      }, {
        "_typename" : "celeritas::ImportPhysicsVector",
        "vector_type" : 2,
        "x" : [1e-4, 100],
        "y" : [0.0919755519795958, 128.588033594672]
      }]
    }
  }, {
    "_typename" : "celeritas::ImportMscModel",
    "particle_pdg" : 11,
    "model_class" : 5,
    "xs_table" : {
      "_typename" : "celeritas::ImportPhysicsTable",
      "table_type" : 4,
      "x_units" : 1,
      "y_units" : 6,
      "physics_vectors" : [{
        "_typename" : "celeritas::ImportPhysicsVector",
        "vector_type" : 2,
        "x" : [100, 100000000],
        "y" : [1.5060677760307e-25, 1.59603068918702e-25]
      }, {
        "_typename" : "celeritas::ImportPhysicsVector",
        "vector_type" : 2,
        "x" : [100, 100000000],
        "y" : [114.932650722669, 116.590357663561]
      }]
    }
  }],
  "sb_data" : [],
  "livermore_pe_data" : [],
  "neutron_elastic_data" : [],
  "atomic_relaxation_data" : [],
  "mu_pair_production_data" : {
    "_typename" : "celeritas::ImportMuPairProductionTable",
    "atomic_number" : [1, 92],
    "physics_vectors" : [{
      "_typename" : "celeritas::ImportPhysics2DVector",
      "x" : [6.74523634948436, 18.4206807439524],
      "y" : [-6.03032981021759, 0],
      "value" : [0, 2.97408672967084e-28, 0, 2.43638436260564e-25]
    }, {
      "_typename" : "celeritas::ImportPhysics2DVector",
      "x" : [6.74523634948436, 18.4206807439524],
      "y" : [-6.03032981021759, 0],
      "value" : [0, 2.39787798860303e-24, 0, 7.93413967608236e-22]
    }]
  },
  "em_params" : {
    "_typename" : "celeritas::ImportEmParameters",
    "energy_loss_fluct" : true,
    "lpm" : true,
    "integral_approach" : true,
    "linear_loss_limit" : 0.01,
    "lowest_electron_energy" : 0.001,
    "lowest_muhad_energy" : 0.001,
    "auger" : false,
    "msc_step_algorithm" : 1,
    "msc_muhad_step_algorithm" : 0,
    "msc_displaced" : 1,
    "msc_muhad_displaced" : 0,
    "msc_range_factor" : 0.04,
    "msc_muhad_range_factor" : 0.2,
    "msc_safety_factor" : 0.6,
    "msc_lambda_limit" : 0.1,
    "msc_theta_limit" : 3.14159265358979,
    "apply_cuts" : false,
    "screening_factor" : 1,
    "angle_limit_factor" : 1,
    "form_factor" : 2
  },
  "trans_params" : {
    "_typename" : "celeritas::ImportTransParameters",
    "looping" : [{"$pair" : "pair<int,celeritas::ImportLoopingThreshold>", "first" : -13, "second" : {
      "_typename" : "celeritas::ImportLoopingThreshold",
      "threshold_trials" : 10,
      "important_energy" : 250
    }}, {"$pair" : "pair<int,celeritas::ImportLoopingThreshold>", "first" : -11, "second" : {
      "_typename" : "celeritas::ImportLoopingThreshold",
      "threshold_trials" : 10,
      "important_energy" : 250
    }}, {"$pair" : "pair<int,celeritas::ImportLoopingThreshold>", "first" : 11, "second" : {
      "_typename" : "celeritas::ImportLoopingThreshold",
      "threshold_trials" : 10,
      "important_energy" : 250
    }}, {"$pair" : "pair<int,celeritas::ImportLoopingThreshold>", "first" : 13, "second" : {
      "_typename" : "celeritas::ImportLoopingThreshold",
      "threshold_trials" : 10,
      "important_energy" : 250
    }}, {"$pair" : "pair<int,celeritas::ImportLoopingThreshold>", "first" : 22, "second" : {
      "_typename" : "celeritas::ImportLoopingThreshold",
      "threshold_trials" : 10,
      "important_energy" : 250
    }}],
    "max_substeps" : 1000
  },
  "optical_params" : {
    "_typename" : "celeritas::ImportOpticalParameters",
    "scintillation_by_particle" : false
  },
  "optical_models" : [],
  "optical_materials" : [],
  "units" : "cgs"
})json",
            str);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
