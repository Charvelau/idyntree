/*
 * Copyright (C) 2015 Fondazione Istituto Italiano di Tecnologia
 * Authors: Silvio Traversaro
 * CopyPolicy: Released under the terms of the LGPLv2.1 or later, see LGPL.TXT
 *
 */

#ifndef IDYNTREE_INVERSE_DYNAMICS_H
#define IDYNTREE_INVERSE_DYNAMICS_H

#include <iDynTree/Model/Indeces.h>

#include <iDynTree/Model/LinkState.h>
#include <iDynTree/Model/JointState.h>

namespace iDynTree
{
    class Model;
    class Traversal;
    class FreeFloatingPos;
    class FreeFloatingVel;
    class FreeFloatingAcc;
    class FreeFloatingGeneralizedTorques;
    class FreeFloatingMassMatrix;
    class JointDOFsDoubleArray;
    class DOFSpatialForceArray;
    class DOFSpatialMotionArray;
    class SpatialMomentum;

    /**
     * \ingroup iDynTreeModel
     *
     * Compute the total linear and angular momentum of a robot, expressed in the world frame.
     *
     * @param[in]  model the used model,
     * @param[in]  linkPositions linkPositions(l) contains the world_H_link transform.
     * @param[in]  linkVels linkVels(l) contains the link l velocity expressed in l frame.
     * @param[out] totalMomentum total momentum, expressed in world frame.
     * @return true if all went well, false otherwise.
     */
    bool ComputeLinearAndAngularMomentum(const Model& model,
                                         const LinkPositions& linkPositions,
                                         const LinkVelArray&  linkVels,
                                               SpatialMomentum& totalMomentum);

    bool RNEADynamicPhase(const Model & model,
                          const Traversal & traversal,
                          const JointPosDoubleArray & jointPos,
                          const LinkVelArray & linksVel,
                          const LinkAccArray & linksAcc,
                          const LinkNetExternalWrenches & linkExtForces,
                                LinkInternalWrenches       & linkIntWrenches,
                                FreeFloatingGeneralizedTorques & baseForceAndJointTorques);

    /**
     * Compute the floating base mass matrix, using the
     * composite rigid body algorithm.
     *
     */
    bool CompositeRigidBodyAlgorithm(const Model& model,
                                     const Traversal& traversal,
                                     const JointPosDoubleArray& jointPos,
                                     LinkCompositeRigidBodyInertias& linkCRBs,
                                     FreeFloatingMassMatrix& massMatrix);


    /**
     * Structure of buffers required by ArticulatedBodyAlgorithm.
     *
     * As the ArticulatedBodyAlgorithm function needs some internal buffers
     * to run, but we don't want to put memory allocation inside the ArticulatedBodyAlgorithm
     * function, we put all the internal buffers in this structure.
     *
     * A convenient resize(Model) function is provided to automatically resize
     * the buffers given a Model.
     */
    struct ArticulatedBodyAlgorithmInternalBuffers
    {
        ArticulatedBodyAlgorithmInternalBuffers() {};

        /**
         * Call resize(model);
         */
        ArticulatedBodyAlgorithmInternalBuffers(const Model & model);

        /**
         * Resize all the buffers to the right size given the model,
         * and reset all the buffers to 0.
         */
        void resize(const Model& model);

        /**
         * Check if the dimension of the buffer is consistent
         * with a model (it should be after a call to resize(model) ).
         */
        bool isConsistent(const Model& model);

        DOFSpatialMotionArray S;
        DOFSpatialForceArray U;
        JointDOFsDoubleArray D;
        JointDOFsDoubleArray u;
        LinkVelArray linksVel;
        LinkAccArray linksBiasAcceleration;
        LinkAccArray linksAccelerations;
        LinkArticulatedBodyInertias linkABIs;
        LinkWrenches linksBiasWrench;

        // Debug quantity
        //LinkWrenches pa;
    };

    /**
     * Compute the floating base acceleration of an unconstrianed
     * robot, using as input the external forces and the joint torques.
     * We follow the algorithm described in Featherstone 2008, modified
     * for the floating base case and for handling fixed joints.
     *
     */
    bool ArticulatedBodyAlgorithm(const Model& model,
                                  const Traversal& traversal,
                                  const FreeFloatingPos& robotPos,
                                  const FreeFloatingVel& robotVel,
                                  const LinkNetExternalWrenches & linkExtWrenches,
                                  const JointDOFsDoubleArray & jointTorques,
                                        ArticulatedBodyAlgorithmInternalBuffers & buffers,
                                        FreeFloatingAcc & robotAcc);



}

#endif
