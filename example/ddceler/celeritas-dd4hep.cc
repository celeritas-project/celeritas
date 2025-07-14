//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/celertias-dd4hep.cc
//---------------------------------------------------------------------------//

#include "DD4hep/DetFactoryHelper.h"
#include "DD4hep/Printout.h"
#include "XML/Layering.h"
#include <cmath>

using namespace std;
using namespace dd4hep;
using namespace dd4hep::detail;

static Ref_t create_detector(Detector& description, xml_h e, SensitiveDetector sens) {
  xml_det_t x_det = e;
  Material material = description.material(x_det.materialStr());
  string det_name = x_det.nameStr();
  string det_type = x_det.typeStr();
  int det_id = x_det.id();

  DetElement detector(det_name, det_id);

  // Create an assembly volume for automatic envelope
  Assembly assembly(det_name + "_assembly");

  // Set visualization attributes
  if (x_det.hasAttr(_U(vis))) {
    assembly.setVisAttributes(description.visAttributes(x_det.visStr()));
  }

  // Create layers
  int layer_num = 0;
  for (xml_coll_t c(x_det, _U(layer)); c; ++c) {
    xml_comp_t x_layer = c;
    layer_num++;

    double layer_rmin = x_layer.inner_r();
    double layer_rmax = x_layer.outer_r();
    double layer_z = x_layer.z_length();

    // Create layer tube
    Tube layer_tube(layer_rmin, layer_rmax, layer_z / 2.0);
    Volume layer_vol(det_name + "_layer_" + to_string(layer_num), layer_tube, material);

    // Set sensitivity if this is an active detector
    if (sens.isValid()) {
      layer_vol.setSensitiveDetector(sens);
    }

    // Create detector element for this layer
    DetElement layer_det(detector, "layer_" + to_string(layer_num), layer_num);

    // Place the layer in the assembly
    PlacedVolume layer_pv = assembly.placeVolume(layer_vol);
    layer_pv.addPhysVolID("layer", layer_num);
    layer_det.setPlacement(layer_pv);

    printout(DEBUG, "SimpleCylindrical",
             "Created layer %d: rmin=%.1f mm, rmax=%.1f mm, z=%.1f mm", layer_num,
             layer_rmin / mm, layer_rmax / mm, layer_z / mm);
  }

  // Create the detector element and place it in the world
  Volume mother_vol = description.pickMotherVolume(detector);
  PlacedVolume detector_pv = mother_vol.placeVolume(assembly);
  detector_pv.addPhysVolID("system", det_id);
  detector.setPlacement(detector_pv);

  printout(INFO, "SimpleCylindrical", "Created detector '%s' with %d layers",
           det_name.c_str(), layer_num);

  return detector;
}

// Register the detector constructor
DECLARE_DETELEMENT(SimpleCylindrical, create_detector)